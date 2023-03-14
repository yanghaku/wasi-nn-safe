/// The type of the elements in a tensor.
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
#[repr(C)]
pub enum TensorType {
    F16,
    F32,
    U8,
    I32,
}

impl TensorType {
    #[inline(always)]
    pub fn byte_size(&self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
            Self::U8 => 1,
            Self::I32 => 4,
        }
    }
}

/// Trait for [`crate::GraphExecutionContext`]'s input and output.
/// Every Object which impl [`ToTensor`] can be used in `set_input` and `get_output`.
///
// wasi-nn will check that ```mul(dimensions()) >= data.len()```, we do not need check it.
pub trait ToTensor {
    fn tensor_type(&self) -> TensorType;

    fn dimensions(&self) -> &[usize];

    /// used in `set_input`
    fn buffer_for_read(&self) -> &[u8];

    /// used in `get_output`
    fn buffer_for_write(&mut self) -> &mut [u8];
}

#[repr(C)]
pub(crate) struct Tensor<'t> {
    pub dimensions: &'t [usize],
    pub tensor_type: TensorType,
    pub data: &'t [u8],
}

impl<'t, T> From<&'t T> for Tensor<'t>
where
    T: ToTensor,
{
    #[inline(always)]
    fn from(t: &'t T) -> Self {
        Self {
            dimensions: t.dimensions(),
            tensor_type: t.tensor_type(),
            data: t.buffer_for_read(),
        }
    }
}

impl<'t> Tensor<'t> {
    #[inline(always)]
    pub fn new(dimensions: &'t [usize], tensor_type: TensorType, data: &'t [u8]) -> Self {
        Self {
            dimensions,
            tensor_type,
            data,
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Tensor, TensorType};

    #[test]
    fn test_enum_tensor_type() {
        assert_eq!(TensorType::F16 as u32, 0);
        assert_eq!(TensorType::F32 as u32, 1);
        assert_eq!(TensorType::U8 as u32, 2);
        assert_eq!(TensorType::I32 as u32, 3);
    }

    #[test]
    fn test_tensor_type_with_wasi_nn() {
        assert_eq!(
            TensorType::F16 as u32,
            wasi_nn::TENSOR_TYPE_F16.raw() as u32
        );
        assert_eq!(
            TensorType::F32 as u32,
            wasi_nn::TENSOR_TYPE_F32.raw() as u32
        );
        assert_eq!(TensorType::U8 as u32, wasi_nn::TENSOR_TYPE_U8.raw() as u32);
        assert_eq!(
            TensorType::I32 as u32,
            wasi_nn::TENSOR_TYPE_I32.raw() as u32
        );
    }

    #[test]
    fn test_tensor_tor_wasi_nn_syscall() {
        assert_eq!(
            std::mem::size_of::<Tensor>(),
            std::mem::size_of::<wasi_nn::Tensor>()
        );
    }
}
