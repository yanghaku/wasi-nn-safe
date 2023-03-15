use crate::error::BackendError;
use crate::tensor::Tensor;
use crate::{Error, GraphEncoding, GraphExecutionTarget};

pub(crate) type GraphHandle = usize;
pub(crate) type GraphExecutionContextHandle = usize;

#[inline(always)]
pub(crate) fn load(
    graph_builder_array: &[&[u8]],
    encoding: GraphEncoding,
    target: GraphExecutionTarget,
) -> Result<GraphHandle, Error> {
    let mut graph_handle = 0;
    let res = unsafe {
        wasi_syscall_inner::load(
            graph_builder_array.as_ptr() as usize,
            graph_builder_array.len(),
            encoding as u32,
            target as u32,
            &mut graph_handle as *mut _ as usize,
        )
    };

    if res == 0 {
        Ok(graph_handle)
    } else {
        Err(Error::BackendError(BackendError::from(res)))
    }
}

#[inline(always)]
pub(crate) fn init_execution_context(
    graph_handle: &GraphHandle,
) -> Result<GraphExecutionContextHandle, Error> {
    let mut ctx_handle = 0;
    let res = unsafe {
        wasi_syscall_inner::init_execution_context(
            *graph_handle,
            &mut ctx_handle as *mut _ as usize,
        )
    };

    if res == 0 {
        Ok(ctx_handle)
    } else {
        Err(Error::BackendError(BackendError::from(res)))
    }
}

#[inline(always)]
pub(crate) fn set_input(
    ctx_handle: &mut GraphExecutionContextHandle,
    index: usize,
    tensor: Tensor,
) -> Result<(), Error> {
    let res =
        unsafe { wasi_syscall_inner::set_input(*ctx_handle, index, &tensor as *const _ as usize) };
    if res == 0 {
        Ok(())
    } else {
        Err(Error::BackendError(BackendError::from(res)))
    }
}

#[inline(always)]
pub(crate) fn compute(ctx_handle: &mut GraphExecutionContextHandle) -> Result<(), Error> {
    let res = unsafe { wasi_syscall_inner::compute(*ctx_handle) };
    if res == 0 {
        Ok(())
    } else {
        Err(Error::BackendError(BackendError::from(res)))
    }
}

#[inline(always)]
pub(crate) fn get_output(
    ctx_handle: &GraphExecutionContextHandle,
    index: usize,
    out_buf: &mut [u8],
) -> Result<usize, Error> {
    let mut out_size = 0;
    let res = unsafe {
        wasi_syscall_inner::get_output(
            *ctx_handle,
            index,
            out_buf.as_mut_ptr() as usize,
            out_buf.len(),
            &mut out_size as *mut _ as usize,
        )
    };

    if res == 0 {
        Ok(out_size)
    } else {
        Err(Error::BackendError(BackendError::from(res)))
    }
}

mod wasi_syscall_inner {
    #[link(wasm_import_module = "wasi_ephemeral_nn")]
    extern "C" {
        pub fn load(
            graph_builder_array_ptr: usize,
            graph_builder_array_len: usize,
            encoding: u32,
            target: u32,
            graph_result_ptr: usize,
        ) -> u32;
        pub fn init_execution_context(graph: usize, context_result_ptr: usize) -> u32;
        pub fn set_input(context: usize, index: usize, tensor_ptr: usize) -> u32;
        pub fn get_output(
            context: usize,
            index: usize,
            out_buffer_ptr: usize,
            out_buffer_max_size: usize,
            result_buffer_size_ptr: usize,
        ) -> u32;
        pub fn compute(context: usize) -> u32;
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_bytes_in_wasm32() {
        assert_eq!(std::mem::size_of::<usize>(), std::mem::size_of::<u32>());
        assert_eq!(std::mem::size_of::<u32>(), std::mem::size_of::<i32>());
        assert_eq!(
            std::mem::size_of::<usize>(),
            std::mem::size_of::<*const u8>()
        );
    }
}
