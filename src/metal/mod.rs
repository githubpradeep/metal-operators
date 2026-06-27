use metal::*;
use std::ffi::c_void;

pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
}

impl MetalContext {
    pub fn new() -> anyhow::Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| anyhow::anyhow!("No Metal device found"))?;
        let queue = device.new_command_queue();
        Ok(Self { device, queue })
    }

    pub fn compile_kernel(&self, source: &str, name: &str) -> anyhow::Result<ComputePipelineState> {
        let options = CompileOptions::new();
        let library = self
            .device
            .new_library_with_source(source, &options)
            .map_err(|e| anyhow::anyhow!("Shader compilation failed: {}", e))?;
        let function = library
            .get_function(name, None)
            .map_err(|e| anyhow::anyhow!("Function '{}' not found: {}", name, e))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| anyhow::anyhow!("Pipeline creation failed: {}", e))?;
        Ok(pipeline)
    }

    pub fn new_buffer<T>(&self, data: &[T]) -> Buffer {
        let ptr = data.as_ptr() as *const c_void;
        let len = (data.len() * std::mem::size_of::<T>()) as u64;
        self.device
            .new_buffer_with_data(ptr, len, MTLResourceOptions::StorageModeShared)
    }

    pub fn new_buffer_uninitialized(&self, byte_size: u64) -> Buffer {
        self.device
            .new_buffer(byte_size, MTLResourceOptions::StorageModeShared)
    }

    pub fn read_buffer<T>(&self, buffer: &Buffer, count: usize) -> Vec<T> {
        let ptr = buffer.contents() as *const T;
        let mut result = Vec::with_capacity(count);
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), count);
            result.set_len(count);
        }
        result
    }
}
