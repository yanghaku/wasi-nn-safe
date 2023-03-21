#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
pub(crate) mod wasi_nn_backend_error;

#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
pub(crate) use wasi_nn_backend_error::BackendError;

/// wasi-nn-safe API error enum
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),

    #[error(
        "Invalid Tensor: Expect data buffer has at least `{expect}` bytes, but it has only `actual` bytes "
    )]
    InvalidTensorError { expect: usize, actual: usize },

    #[error("Backend Error: {0}")]
    BackendError(#[from] BackendError),
}
