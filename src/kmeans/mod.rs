use crate::metal::MetalContext;
use metal::*;

const SHADER_SRC: &str = include_str!("../../shaders/kmeans.metal");

pub struct KMeansConfig {
    pub k: usize,
    pub max_iterations: usize,
    pub tolerance: f32,
    pub seed: u64,
    pub init_centroids: Option<Vec<f32>>,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self { k: 8, max_iterations: 100, tolerance: 1e-4, seed: 42, init_centroids: None }
    }
}

pub struct KMeans {
    config: KMeansConfig,
    centroids: Vec<f32>,
    inertia: f32,
    n_iter: usize,
    labels: Vec<usize>,
}

impl KMeans {
    pub fn new(config: KMeansConfig) -> Self {
        Self {
            config,
            centroids: Vec::new(),
            inertia: 0.0,
            n_iter: 0,
            labels: Vec::new(),
        }
    }

    pub fn centroids(&self) -> &[f32] { &self.centroids }
    pub fn labels(&self) -> &[usize] { &self.labels }
    pub fn inertia(&self) -> f32 { self.inertia }
    pub fn n_iter(&self) -> usize { self.n_iter }

    pub fn fit(
        &mut self,
        ctx: &MetalContext,
        data: &[f32],
        n: usize,
        d: usize,
    ) -> anyhow::Result<()> {
        if d == 0 || n == 0 || self.config.k == 0 || self.config.k > n {
            anyhow::bail!("Invalid parameters: n={}, d={}, k={}", n, d, self.config.k);
        }
        if data.len() != n * d {
            anyhow::bail!("Data length mismatch: expected {}, got {}", n * d, data.len());
        }
        if let Some(ref init) = self.config.init_centroids {
            if init.len() != self.config.k * d {
                anyhow::bail!(
                    "init_centroids length {} mismatch: expected {} (k={}, d={})",
                    init.len(), self.config.k * d, self.config.k, d
                );
            }
        }

        let (kernel_name, kernel) = pick_assign_kernel(self.config.k, d);
        let pipeline_assign = ctx.compile_kernel(SHADER_SRC, kernel_name)?;

        let point_buffer = ctx.new_buffer(data);
        let assign_buffer = ctx.new_buffer_uninitialized((n * std::mem::size_of::<u32>()) as u64);
        let dist_buffer = ctx.new_buffer_uninitialized((n * std::mem::size_of::<f32>()) as u64);

        // precompute point norms for kernels that use them (simdgroup, split-D)
        let use_norms = kernel_uses_norms(&kernel);
        let norms_x = if use_norms { Some(compute_norms(data, n, d)) } else { None };
        let norms_x_buf = norms_x.as_ref().map(|x| ctx.new_buffer(x));

        self.centroids = match &self.config.init_centroids {
            Some(c) => c.clone(),
            None => self.init_kmeans_plusplus(ctx, data, n, d)?,
        };

        // Compile centroid tiled pipeline
        let pipeline_centroid = ctx.compile_kernel(SHADER_SRC, "kmeans_centroid_tiled")?;

        let (groups, tg_size) = match &kernel {
            AssignKernel::Naive => dispatch_naive(n),
            AssignKernel::Simdgroup | AssignKernel::SimdgroupC16 => dispatch_simdgroup(n),
            AssignKernel::SplitD => dispatch_splitd(n),
        };

        for iter in 0..self.config.max_iterations {
            let t0 = std::time::Instant::now();

            let centroids_buf = ctx.new_buffer(&self.centroids);
            let norms_c_buf = if use_norms {
                Some(ctx.new_buffer(&compute_norms(&self.centroids, self.config.k, d)))
            } else {
                None
            };
            let t_buf = t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = std::time::Instant::now();
            dispatch_assign(
                ctx, &pipeline_assign,
                &point_buffer, &centroids_buf,
                &assign_buffer, &dist_buffer,
                norms_x_buf.as_ref(), norms_c_buf.as_ref(),
                n, self.config.k, d, groups, tg_size,
                &kernel,
            );
            let t_assign = t1.elapsed().as_secs_f64() * 1000.0;

            // ── GPU centroid update via tiled threadgroup-atomic kernel ──
            let t_cent_start = std::time::Instant::now();
            let k = self.config.k;
            let shared_needed = (k * d + k) * 4;

            let (max_shift, via_gpu) = if shared_needed <= 32_768 {
                let sums_init = vec![0.0f32; k * d];
                let counts_init = vec![0i32; k];
                let sums_buf = ctx.new_buffer(&sums_init);
                let counts_buf = ctx.new_buffer(&counts_init);

                let cmd_buf = ctx.queue.new_command_buffer();
                let enc = cmd_buf.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline_centroid);
                enc.set_buffer(0, Some(&point_buffer), 0);
                enc.set_buffer(1, Some(&assign_buffer), 0);
                enc.set_buffer(2, Some(&sums_buf), 0);
                enc.set_buffer(3, Some(&counts_buf), 0);
                set_uint(&enc, 4, n as u32);
                set_uint(&enc, 5, k as u32);
                set_uint(&enc, 6, d as u32);
                enc.set_threadgroup_memory_length(0, shared_needed as u64);
                let ptile: u64 = 128;
                let tg = MTLSize { width: ptile, height: 1, depth: 1 };
                let grp = MTLSize { width: ((n as u64 + ptile - 1) / ptile), height: 1, depth: 1 };
                enc.dispatch_thread_groups(grp, tg);
                enc.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();

                let sums: Vec<f32> = ctx.read_buffer(&sums_buf, k * d);
                let counts: Vec<i32> = ctx.read_buffer(&counts_buf, k);

                let mut new_centroids = Vec::with_capacity(k * d);
                for c in 0..k {
                    let cnt = counts[c];
                    if cnt > 0 {
                        let inv = 1.0f32 / cnt as f32;
                        for dim in 0..d {
                            new_centroids.push(sums[c * d + dim] * inv);
                        }
                    } else {
                        for _dim in 0..d {
                            new_centroids.push(0.0f32);
                        }
                    }
                }
                let shift = max_centroid_shift(&self.centroids, &new_centroids, k, d);
                self.centroids = new_centroids;
                (shift, true)
            } else {
                let assignments: Vec<u32> = ctx.read_buffer(&assign_buffer, n);
                let new_centroids = Self::compute_centroids(data, n, d, k, &assignments);
                let shift = max_centroid_shift(&self.centroids, &new_centroids, k, d);
                self.centroids = new_centroids;
                (shift, false)
            };

            let t_cent = t_cent_start.elapsed().as_secs_f64() * 1000.0;
            self.n_iter = iter + 1;

            if iter == 0 {
                eprintln!("  timing[{}]: buf={:.3}ms assign={:.3}ms cent_{}={:.3}ms total={:.3}ms",
                    iter, t_buf, t_assign, if via_gpu { "gpu" } else { "cpu" }, t_cent,
                    t0.elapsed().as_secs_f64() * 1000.0);
            }

            if max_shift < self.config.tolerance {
                break;
            }
        }

        let assignments: Vec<u32> = ctx.read_buffer(&assign_buffer, n);
        self.labels = assignments.into_iter().map(|x| x as usize).collect();
        self.compute_inertia(data, n, d);
        Ok(())
    }

    pub fn predict(
        &self, ctx: &MetalContext, data: &[f32], n: usize, d: usize,
    ) -> anyhow::Result<Vec<usize>> {
        let (kernel_name, kernel) = pick_assign_kernel(self.config.k, d);
        let pipeline = ctx.compile_kernel(SHADER_SRC, kernel_name)?;
        let point_buffer = ctx.new_buffer(data);
        let centroids_buffer = ctx.new_buffer(&self.centroids);
        let assign_buffer = ctx.new_buffer_uninitialized((n * std::mem::size_of::<u32>()) as u64);
        let dist_buffer = ctx.new_buffer_uninitialized((n * std::mem::size_of::<f32>()) as u64);

        let use_norms = kernel_uses_norms(&kernel);
        let norms_x = if use_norms { Some(compute_norms(data, n, d)) } else { None };
        let norms_x_buf = norms_x.as_ref().map(|x| ctx.new_buffer(x));
        let norms_c_buf = if use_norms {
            Some(ctx.new_buffer(&compute_norms(&self.centroids, self.config.k, d)))
        } else {
            None
        };

        let (groups, tg_size) = match &kernel {
            AssignKernel::Naive => dispatch_naive(n),
            AssignKernel::Simdgroup | AssignKernel::SimdgroupC16 => dispatch_simdgroup(n),
            AssignKernel::SplitD => dispatch_splitd(n),
        };
        dispatch_assign(
            ctx, &pipeline,
            &point_buffer, &centroids_buffer,
            &assign_buffer, &dist_buffer,
            norms_x_buf.as_ref(), norms_c_buf.as_ref(),
            n, self.config.k, d, groups, tg_size,
            &kernel,
        );

        let assignments: Vec<u32> = ctx.read_buffer(&assign_buffer, n);
        Ok(assignments.into_iter().map(|x| x as usize).collect())
    }

    fn compute_inertia(&mut self, data: &[f32], n: usize, d: usize) {
        self.inertia = 0.0;
        for i in 0..n {
            let label = self.labels[i];
            let mut dist = 0.0f32;
            for dim in 0..d {
                let diff = data[i * d + dim] - self.centroids[label * d + dim];
                dist += diff * diff;
            }
            self.inertia += dist;
        }
    }

    #[allow(dead_code)]
    fn compute_centroids(data: &[f32], n: usize, d: usize, k: usize, assignments: &[u32]) -> Vec<f32> {
        let mut sums = vec![0.0f32; k * d];
        let mut counts = vec![0u64; k];
        for i in 0..n {
            let label = assignments[i] as usize;
            counts[label] += 1;
            for dim in 0..d {
                sums[label * d + dim] += data[i * d + dim];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for dim in 0..d {
                    sums[c * d + dim] *= inv;
                }
            }
        }
        sums
    }

    fn init_kmeans_plusplus(
        &self, ctx: &MetalContext, data: &[f32], n: usize, d: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let seed = if self.config.seed == 0 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        } else {
            self.config.seed
        };
        let mut rng = fastrand::Rng::with_seed(seed);

        let mut centroids = Vec::with_capacity(self.config.k * d);
        let first_idx = rng.usize(0..n);
        centroids.extend_from_slice(&data[first_idx * d..(first_idx + 1) * d]);

        if self.config.k == 1 {
            return Ok(centroids);
        }

        let pipeline = ctx.compile_kernel(SHADER_SRC, "kmeans_compute_min_distances")?;
        let point_buffer = ctx.new_buffer(data);
        let dist_buffer = ctx.new_buffer_uninitialized((n * std::mem::size_of::<f32>()) as u64);
        let (groups, tg_size) = dispatch_naive(n);

        for c in 1..self.config.k {
            let centroids_buffer = ctx.new_buffer(&centroids);
            let num_centroids = c as u32;

            let cmd_buffer = ctx.queue.new_command_buffer();
            let encoder = cmd_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&point_buffer), 0);
            encoder.set_buffer(1, Some(&centroids_buffer), 0);
            encoder.set_buffer(2, Some(&dist_buffer), 0);
            set_uint(encoder, 3, n as u32);
            set_uint(encoder, 4, num_centroids);
            set_uint(encoder, 5, d as u32);
            encoder.dispatch_thread_groups(groups, tg_size);
            encoder.end_encoding();
            cmd_buffer.commit();
            cmd_buffer.wait_until_completed();

            let dists: Vec<f32> = ctx.read_buffer(&dist_buffer, n);

            let total_weight: f32 = dists.iter().sum();
            if total_weight == 0.0 {
                let pick = rng.usize(0..n);
                centroids.extend_from_slice(&data[pick * d..(pick + 1) * d]);
                continue;
            }

            let threshold = rng.f32() * total_weight;
            let mut cumulative = 0.0f32;
            let mut pick_idx = n - 1;
            for (i, &dist) in dists.iter().enumerate() {
                cumulative += dist;
                if cumulative >= threshold {
                    pick_idx = i;
                    break;
                }
            }
            centroids.extend_from_slice(&data[pick_idx * d..(pick_idx + 1) * d]);
        }

        Ok(centroids)
    }
}

// ── helpers ──────────────────────────────────────────────────

enum AssignKernel {
    Naive,
    Simdgroup,
    SimdgroupC16,
    SplitD,
}

fn pick_assign_kernel(k: usize, d: usize) -> (&'static str, AssignKernel) {
    if k == 0 { return ("kmeans_assign", AssignKernel::Naive); }

    if d >= 8 && d % 8 == 0 {
        let dim_tiles = (d + 7) / 8;

        // try CTILE=16 when K ≤ 16 — single centroid tile, fewer barriers
        if k <= 16 {
            let shared_c16 = (24 * d + dim_tiles * 128 + 16) * 4;
            if shared_c16 <= 32_768 {
                return ("kmeans_assign_simdgroup_c16", AssignKernel::SimdgroupC16);
            }
        }

        // default CTILE=8 simdgroup kernel
        let shared_c8 = (16 * d + dim_tiles * 64 + 16) * 4;
        if shared_c8 <= 32_768 {
            return ("kmeans_assign_simdgroup", AssignKernel::Simdgroup);
        }
    }

    if d > 0 {
        ("kmeans_assign_splitd", AssignKernel::SplitD)
    } else {
        ("kmeans_assign", AssignKernel::Naive)
    }
}

fn kernel_uses_norms(kernel: &AssignKernel) -> bool {
    matches!(kernel, AssignKernel::Simdgroup | AssignKernel::SimdgroupC16 | AssignKernel::SplitD)
}

fn compute_norms(data: &[f32], n: usize, d: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let row = &data[i * d..(i + 1) * d];
            row.iter().map(|x| x * x).sum::<f32>()
        })
        .collect()
}

fn dispatch_splitd(n: usize) -> (MTLSize, MTLSize) {
    const PTILE: u64 = 128;
    let groups = MTLSize { width: ((n as u64 + PTILE - 1) / PTILE), height: 1, depth: 1 };
    let threads = MTLSize { width: PTILE, height: 1, depth: 1 };
    (groups, threads)
}

fn dispatch_simdgroup(n: usize) -> (MTLSize, MTLSize) {
    let groups = MTLSize { width: ((n as u64 + 7) / 8), height: 1, depth: 1 };
    let threads = MTLSize { width: 128, height: 1, depth: 1 };
    (groups, threads)
}

fn dispatch_naive(n: usize) -> (MTLSize, MTLSize) {
    const TG: u64 = 256;
    let groups = MTLSize { width: ((n as u64 + TG - 1) / TG), height: 1, depth: 1 };
    let threads = MTLSize { width: TG, height: 1, depth: 1 };
    (groups, threads)
}

fn set_uint(encoder: &ComputeCommandEncoderRef, index: u64, value: u32) {
    let len = std::mem::size_of::<u32>() as u64;
    encoder.set_bytes(index, len, std::ptr::from_ref(&value).cast());
}

fn encode_assign(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    point_buffer: &Buffer,
    centroids_buffer: &Buffer,
    assign_buffer: &Buffer,
    dist_buffer: &Buffer,
    norms_x_buf: Option<&Buffer>,
    norms_c_buf: Option<&Buffer>,
    n: usize, k: usize, d: usize,
    groups: MTLSize, tg_size: MTLSize,
    kernel: &AssignKernel,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(point_buffer), 0);
    encoder.set_buffer(1, Some(centroids_buffer), 0);
    encoder.set_buffer(2, Some(assign_buffer), 0);
    encoder.set_buffer(3, Some(dist_buffer), 0);

    match kernel {
        AssignKernel::Naive => {
            set_uint(&encoder, 4, n as u32);
            set_uint(&encoder, 5, k as u32);
            set_uint(&encoder, 6, d as u32);
        }
        AssignKernel::Simdgroup | AssignKernel::SimdgroupC16 | AssignKernel::SplitD => {
            encoder.set_buffer(4, norms_x_buf.map(|v| &**v), 0);
            encoder.set_buffer(5, norms_c_buf.map(|v| &**v), 0);
            set_uint(&encoder, 6, n as u32);
            set_uint(&encoder, 7, k as u32);
            set_uint(&encoder, 8, d as u32);
            set_uint(&encoder, 9, tg_size.width as u32);

            let shared_bytes = match kernel {
                AssignKernel::Simdgroup => {
                    let dim_tiles = (d + 7) / 8;
                    (16 * d + dim_tiles * 64 + 16) as u64 * 4
                }
                AssignKernel::SimdgroupC16 => {
                    let dim_tiles = (d + 7) / 8;
                    (24 * d + dim_tiles * 128 + 16) as u64 * 4
                }
                AssignKernel::SplitD => {
                    // BD × CTILE = 32 × 8 = 256 floats
                    256u64 * 4
                }
                _ => unreachable!(),
            };
            encoder.set_threadgroup_memory_length(0, shared_bytes);
        }
    }

    encoder.dispatch_thread_groups(groups, tg_size);
}

fn dispatch_assign(
    ctx: &MetalContext,
    pipeline: &ComputePipelineState,
    point_buffer: &Buffer,
    centroids_buffer: &Buffer,
    assign_buffer: &Buffer,
    dist_buffer: &Buffer,
    norms_x_buf: Option<&Buffer>,
    norms_c_buf: Option<&Buffer>,
    n: usize, k: usize, d: usize,
    groups: MTLSize, tg_size: MTLSize,
    kernel: &AssignKernel,
) {
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encode_assign(
        &encoder, pipeline,
        point_buffer, centroids_buffer,
        assign_buffer, dist_buffer,
        norms_x_buf, norms_c_buf,
        n, k, d, groups, tg_size, kernel,
    );
    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();
}

fn max_centroid_shift(old: &[f32], new: &[f32], k: usize, d: usize) -> f32 {
    let mut max_shift = 0.0f32;
    for c in 0..k {
        for dim in 0..d {
            let shift = (new[c * d + dim] - old[c * d + dim]).abs();
            if shift > max_shift { max_shift = shift; }
        }
    }
    max_shift
}
