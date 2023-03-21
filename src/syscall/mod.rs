#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
mod wasm32_wasi;

#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
use wasm32_wasi as syscall;

pub(crate) use syscall::{compute, get_output, init_execution_context, load, set_input};
pub(crate) use syscall::{GraphExecutionContextHandle, GraphHandle};
