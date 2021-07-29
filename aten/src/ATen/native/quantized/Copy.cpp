#include <ATen/native/quantized/Copy.h>

#include <ATen/ATen.h>
#include <ATen/native/quantized/affine_quantizer.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {

// Copying from float to QInt, used for assigning float value to QTensor
Tensor& quantized_copy_from_float_cpu_(Tensor& self, const Tensor& src) {
  TORCH_CHECK(
      src.scalar_type() == at::kFloat,
      "Quantized copy only works with kFloat as source Tensor");
  TORCH_CHECK(
      self.is_contiguous() && src.is_contiguous(),
      "Quantized copy only works with contiguous Tensors");
  TORCH_CHECK(
      self.sizes().equals(src.sizes()),
      "Quantized copy only works with Tensors with the same shape");
  TORCH_CHECK(
      self.device().type() == kCPU,
      "Quantized copy only works with QuantizedCPU Tensors");
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "Copy", [&]() {
    float* src_data = src.data_ptr<float>();
    scalar_t* self_data = self.data_ptr<scalar_t>();
    for (int i = 0; i < self.numel(); ++i) {
      self_data[i] = quantize_val<scalar_t>(
          self.q_scale(), self.q_zero_point(), src_data[i]);
    }
  });
  return self;
}

static C10_ALWAYS_INLINE Tensor& copy_quantized_for_key_(
    DispatchKey include_key, DispatchKey exclude_key, Tensor & self,
    const Tensor & src, bool non_blocking)  {

  TORCH_CHECK(self.is_quantized(), "Copying to non-quantized Tensor"
                                   "is not allowed in this function");
  if(src.is_quantized())  {
    TORCH_CHECK(self.qscheme() == src.qscheme(),
                "Quantized Copy only works with same qscheme");
    TORCH_CHECK(self.scalar_type() == src.scalar_type());
    set_quantizer_(self, src.quantizer());
    c10::impl::IncludeDispatchKeyGuard include_quantized_guard(include_key);
    c10::impl::ExcludeDispatchKeyGuard exclude_quantized_guard(exclude_key);
    return self.copy_(src, non_blocking);  // redispatch!
  }
  else {
    return quantized_copy_from_float_cpu_(self, src);
  }
}

Tensor & copy_quantized_cpu_(Tensor &self, const Tensor &src, bool non_blocking) {
  return copy_quantized_for_key_(DispatchKey::CPU,
      DispatchKey::QuantizedCPU, self, src, non_blocking);
}

Tensor & copy_quantized_cuda_(Tensor &self, const Tensor &src, bool non_blocking) {
  return copy_quantized_for_key_(DispatchKey::CUDA,
      DispatchKey::QuantizedCUDA, self, src, non_blocking);
}

Tensor & copy_quantized_xpu_(Tensor &self, const Tensor &src, bool non_blocking) {
  return copy_quantized_for_key_(DispatchKey::XPU,
      DispatchKey::QuantizedXPU, self, src, non_blocking);
}
} // namespace native
} // namespace at
