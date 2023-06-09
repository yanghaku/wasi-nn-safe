#[derive(thiserror::Error, Debug)]
pub enum BackendError {
    #[error("WASI-NN Backend Error: Caller module passed an invalid argument")]
    InvalidArgument,
    #[error("WASI-NN Backend Error: Invalid Encoding")]
    InvalidEncoding,
    #[error("WASI-NN Backend Error: Caller module is missing a memory export")]
    MissingMemory,
    #[error("WASI-NN Backend Error: Device or resource busy")]
    Busy,
    #[error("WASI-NN Backend Error: Runtime Error")]
    RuntimeError,
    #[error("Unknown Wasi-NN Backend Error Code `{0}`")]
    UnknownError(u32),
}

impl BackendError {
    #[inline(always)]
    pub(crate) fn from(value: u32) -> Self {
        match value {
            1 => Self::InvalidArgument,
            2 => Self::InvalidEncoding,
            3 => Self::MissingMemory,
            4 => Self::Busy,
            5 => Self::RuntimeError,
            _ => Self::UnknownError(value),
        }
    }
}

#[cfg(test)]
mod test {
    use super::BackendError;

    macro_rules! test_enum_eq {
        ( $v:expr, $enum_name:ident, $enum_element:ident ) => {
            match $enum_name::from($v) {
                $enum_name::$enum_element => {}
                _ => {
                    assert!(false);
                }
            }
        };
    }

    #[test]
    fn test_wasi_nn_backend_error_from_u32() {
        test_enum_eq!(1, BackendError, InvalidArgument);
        test_enum_eq!(2, BackendError, InvalidEncoding);
        test_enum_eq!(3, BackendError, MissingMemory);
        test_enum_eq!(4, BackendError, Busy);
        test_enum_eq!(5, BackendError, RuntimeError);
    }

    #[test]
    fn test_backend_error_with_wasi_nn() {
        test_enum_eq!(
            wasi_nn::NN_ERRNO_INVALID_ARGUMENT.raw() as u32,
            BackendError,
            InvalidArgument
        );
        test_enum_eq!(
            wasi_nn::NN_ERRNO_INVALID_ENCODING.raw() as u32,
            BackendError,
            InvalidEncoding
        );
        test_enum_eq!(
            wasi_nn::NN_ERRNO_MISSING_MEMORY.raw() as u32,
            BackendError,
            MissingMemory
        );
        test_enum_eq!(wasi_nn::NN_ERRNO_BUSY.raw() as u32, BackendError, Busy);
        test_enum_eq!(
            wasi_nn::NN_ERRNO_RUNTIME_ERROR.raw() as u32,
            BackendError,
            RuntimeError
        );
    }
}
