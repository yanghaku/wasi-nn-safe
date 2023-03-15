#[cfg_attr(
    all(target_arch = "wasm32", target_os = "wasi"),
    path = "wasm32_wasi.rs"
)]
mod syscall;

pub(crate) use syscall::{compute, get_output, init_execution_context, load, set_input};
pub(crate) use syscall::{GraphExecutionContextHandle, GraphHandle};
