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

        let (kernel_name, use_simdgroup) = pick_assign_kernel(self.config.k, d);
        let pipeline_assign = ctx.compile_kernel(SHADER_SRC, kernel_name)?;

        let point_buffer = ctx.new_buffer(data);
        let assign_buffer = ctx.new_buffer_uninitialized((n * std::mem::size_of::<u32>()) as u64);
        let dist_buffer = ctx.new_buffer_uninitialized((n * std::mem::size_of::<f32>()) as u64);

        // precompute point norms once for the simdgroup kernel
        let norms_x = if use_simdgroup { Some(compute_norms(data, n, d)) } else { None };
        let norms_x_buf = norms_x.as_ref().map(|x| ctx.new_buffer(x));

        self.centroids = match &self.config.init_centroids {
            Some(c) => c.clone(),
            None => self.init_kmeans_plusplus(ctx, data, n, d)?,
        };

        let mut old_centroids = vec![0.0f32; self.config.k * d];
        let (groups, tg_size) = dispatch_size(n, 128);

        for iter in 0..self.config.max_iterations {
            old_centroids.copy_from_slice(&self.centroids);
            let centroids_buffer = ctx.new_buffer(&self.centroids);

            let norms_c_buf = if use_simdgroup {
                Some(ctx.new_buffer(&compute_norms(&self.centroids, self.config.k, d)))
            } else {
                None
            };

            dispatch_assign(
                ctx, &pipeline_assign,
                &point_buffer, &centroids_buffer,
                &assign_buffer, &dist_buffer,
                norms_x_buf.as_ref(), norms_c_buf.as_ref(),
                n, self.config.k, d, groups, tg_size,
                use_simdgroup,
            );

            let assignments: Vec<u32> = ctx.read_buffer(&assign_buffer, n);
            let new_centroids = Self::compute_centroids(data, n, d, self.config.k, &assignments);

            let max_shift = max_centroid_shift(&self.centroids, &new_centroids, self.config.k, d);
            self.centroids = new_centroids;
            self.n_iter = iter + 1;

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
        let (kernel_name, use_simdgroup) = pick_assign_kernel(self.config.k, d);
        let pipeline = ctx.compile_kernel(SHADER_SRC, kernel_name)?;
        let point_buffer = ctx.new_buffer(data);
        let centroids_buffer = ctx.new_buffer(&self.centroids);
        let assign_buffer = ctx.new_buffer_uninitialized((n * std::mem::size_of::<u32>()) as u64);
        let dist_buffer = ctx.new_buffer_uninitialized((n * std::mem::size_of::<f32>()) as u64);

        let norms_x = if use_simdgroup { Some(compute_norms(data, n, d)) } else { None };
        let norms_x_buf = norms_x.as_ref().map(|x| ctx.new_buffer(x));
        let norms_c_buf = if use_simdgroup {
            Some(ctx.new_buffer(&compute_norms(&self.centroids, self.config.k, d)))
        } else {
            None
        };

        let (groups, tg_size) = dispatch_size(n, 128);
        dispatch_assign(
            ctx, &pipeline,
            &point_buffer, &centroids_buffer,
            &assign_buffer, &dist_buffer,
            norms_x_buf.as_ref(), norms_c_buf.as_ref(),
            n, self.config.k, d, groups, tg_size,
            use_simdgroup,
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
        let (groups, tg_size) = dispatch_size(n, 256);

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

fn pick_assign_kernel(k: usize, d: usize) -> (&'static str, bool) {
    if d < 8 || d % 8 != 0 || k == 0 { return ("kmeans_assign", false); }
    let num_tiles = (d + 7) / 8;
    // sh_pts(8d) + sh_cent(8d) + sh_dots(num_tiles*64) + sh_best_dist(8) + sh_best_lbl(8)
    let shared = (16 * d + num_tiles * 64 + 16) * 4;
    if shared <= 32_768 { ("kmeans_assign_simdgroup", true) }
    else { ("kmeans_assign", false) }
}

fn compute_norms(data: &[f32], n: usize, d: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let row = &data[i * d..(i + 1) * d];
            row.iter().map(|x| x * x).sum::<f32>()
        })
        .collect()
}

fn dispatch_size(n: usize, tg_size: u64) -> (MTLSize, MTLSize) {
    // simdgroup kernel works in tiles of 8 points per threadgroup
    let actual_tg = if tg_size >= 128 { tg_size } else { 128 };
    let groups = MTLSize { width: ((n as u64 + 7) / 8), height: 1, depth: 1 };
    let threads = MTLSize { width: actual_tg, height: 1, depth: 1 };
    (groups, threads)
}

fn set_uint(encoder: &ComputeCommandEncoderRef, index: u64, value: u32) {
    let len = std::mem::size_of::<u32>() as u64;
    encoder.set_bytes(index, len, std::ptr::from_ref(&value).cast());
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
    use_simdgroup: bool,
) {
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(point_buffer), 0);
    encoder.set_buffer(1, Some(centroids_buffer), 0);
    encoder.set_buffer(2, Some(assign_buffer), 0);
    encoder.set_buffer(3, Some(dist_buffer), 0);

    if use_simdgroup {
        encoder.set_buffer(4, norms_x_buf.map(|v| &**v), 0);
        encoder.set_buffer(5, norms_c_buf.map(|v| &**v), 0);
        set_uint(&encoder, 6, n as u32);
        set_uint(&encoder, 7, k as u32);
        set_uint(&encoder, 8, d as u32);
        set_uint(&encoder, 9, tg_size.width as u32);

        let num_tiles = (d + 7) / 8;
        let shared_bytes = (16 * d + num_tiles * 64 + 16) as u64 * 4;
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    } else {
        set_uint(&encoder, 4, n as u32);
        set_uint(&encoder, 5, k as u32);
        set_uint(&encoder, 6, d as u32);
    }

    encoder.dispatch_thread_groups(groups, tg_size);
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
