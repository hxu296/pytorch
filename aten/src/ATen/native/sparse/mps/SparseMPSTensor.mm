#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_coalesce_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/zeros_native.h>
#endif

#include <ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>
#include <ATen/ops/_validate_compressed_sparse_indices.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/diff.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/repeat_interleave.h>
#include <ATen/ops/repeat_interleave_native.h>
#include <ATen/ops/remainder.h>

#include <array>
#include <functional>
#include <numeric>
#include <optional>

namespace at::native {

using namespace mps;
using namespace at::sparse;

void _validate_compressed_sparse_indices_mps(
    bool is_crow,
    const Tensor& cidx,
    const Tensor& idx,
    int64_t cdim,
    int64_t dim,
    int64_t nnz);

namespace mps {
namespace csr {

static void build_batch_ptr_mps_out(
    const Tensor& batch_indices,
    int64_t batch_count,
    const Tensor& batch_ptr) {
  TORCH_CHECK(
      batch_indices.is_mps() && batch_ptr.is_mps(),
      "build_batch_ptr_mps_out: expected MPS tensors");
  TORCH_CHECK(
      batch_ptr.scalar_type() == kLong,
      "build_batch_ptr_mps_out: expected output dtype int64 but got ",
      batch_ptr.scalar_type());
  TORCH_CHECK(
      batch_ptr.numel() == batch_count + 1,
      "build_batch_ptr_mps_out: expected output shape [",
      batch_count + 1,
      "] but got ",
      batch_ptr.numel());

  batch_ptr.zero_();

  if (batch_count == 0 || batch_indices.numel() == 0) {
    return;
  }

  auto options = batch_ptr.options();
  Tensor counts = at::empty({batch_count}, options);
  counts.zero_();
  Tensor ones = at::empty(batch_indices.sizes(), options);
  ones.fill_(1);
  counts.scatter_add_(0, batch_indices, ones);

  Tensor prefix = at::cumsum(counts, /*dim=*/0);
  batch_ptr.slice(/*dim=*/0, /*start=*/1, /*end=*/batch_count + 1).copy_(prefix);
}

static Tensor build_batch_ptr_mps(const Tensor& batch_indices, int64_t batch_count) {
  auto options = batch_indices.options().dtype(kLong);
  Tensor batch_ptr = at::empty({batch_count + 1}, options);
  batch_ptr.zero_();
  build_batch_ptr_mps_out(batch_indices, batch_count, batch_ptr);
  return batch_ptr;
}

static void build_row_ptr_per_batch_mps_out(
    const Tensor& rows,
    const Tensor& batch_ptr,
    int64_t batch_count,
    int64_t rows_per_batch,
    const Tensor& row_ptr) {
  TORCH_CHECK(
      rows.is_mps() && batch_ptr.is_mps() && row_ptr.is_mps(),
      "build_row_ptr_per_batch_mps_out: expected MPS tensors");
  TORCH_CHECK(
      batch_ptr.scalar_type() == kLong,
      "build_row_ptr_per_batch_mps_out: expected batch_ptr dtype int64 but got ",
      batch_ptr.scalar_type());
  TORCH_CHECK(
      row_ptr.scalar_type() == kLong,
      "build_row_ptr_per_batch_mps_out: expected row_ptr dtype int64 but got ",
      row_ptr.scalar_type());
  TORCH_CHECK(
      row_ptr.numel() == batch_count * (rows_per_batch + 1),
      "build_row_ptr_per_batch_mps_out: expected output shape [",
      batch_count * (rows_per_batch + 1),
      "] but got ",
      row_ptr.numel());

  row_ptr.zero_();

  const auto nnz = rows.numel();
  if (nnz == 0 || rows_per_batch == 0 || batch_count == 0) {
    return;
  }

  Tensor batch_lengths = at::diff(batch_ptr);
  TORCH_INTERNAL_ASSERT(batch_lengths.numel() == batch_count);

  Tensor batch_ids = at::_ops::repeat_interleave_self_Tensor::call(
      at::arange(batch_count, rows.options()),
      batch_lengths.to(rows.scalar_type()),
      /*dim=*/0,
      ::std::optional<int64_t>{});
  Tensor flat_indices = batch_ids.mul(rows_per_batch);
  flat_indices.add_(rows);
  Tensor counts_flat = at::empty({batch_count * rows_per_batch}, row_ptr.options());
  counts_flat.zero_();
  Tensor ones_flat = at::empty(rows.sizes(), row_ptr.options());
  ones_flat.fill_(1);
  counts_flat.scatter_add_(0, flat_indices, ones_flat);

  Tensor counts = counts_flat.view({batch_count, rows_per_batch});
  Tensor row_ptr_view = row_ptr.view({batch_count, rows_per_batch + 1});
  row_ptr_view.slice(/*dim=*/1, /*start=*/1, /*end=*/rows_per_batch + 1).copy_(
      at::cumsum(counts, /*dim=*/1));
}

static Tensor build_row_ptr_per_batch_mps(
    const Tensor& rows,
    const Tensor& batch_ptr,
    int64_t batch_count,
    int64_t rows_per_batch) {
  auto options = rows.options().dtype(kLong);
  Tensor row_ptr = at::empty({batch_count * (rows_per_batch + 1)}, options);
  build_row_ptr_per_batch_mps_out(rows, batch_ptr, batch_count, rows_per_batch, row_ptr);
  return row_ptr;
}

static void expand_csr_rows_to_coo_out(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    int64_t rows_per_batch,
    bool out_int32,
    bool transpose,
    const Tensor& coo_indices) {
  TORCH_CHECK(
      crow_indices.is_mps() && col_indices.is_mps() && coo_indices.is_mps(),
      "expand_csr_rows_to_coo: expected MPS tensors");
  TORCH_CHECK(
      crow_indices.scalar_type() == kLong,
      "expand_csr_rows_to_coo: crow_indices must be int64");
  TORCH_CHECK(
      col_indices.scalar_type() == (out_int32 ? kInt : kLong),
      "expand_csr_rows_to_coo: col_indices dtype mismatch");

  TORCH_CHECK(crow_indices.dim() >= 1, "expand_csr_rows_to_coo: expected batched crow_indices");

  if (col_indices.numel() == 0) {
    coo_indices.zero_();
    return;
  }

  const int64_t rows_plus_one = crow_indices.size(-1);
  TORCH_CHECK(
      rows_plus_one == rows_per_batch + 1,
      "expand_csr_rows_to_coo: crow_indices last dimension must equal rows_per_batch + 1");

  auto batch_shape = crow_indices.sizes().slice(0, crow_indices.dim() - 1);
  const int64_t batch_size = std::accumulate(
      batch_shape.begin(), batch_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());

  TORCH_CHECK(
      col_indices.numel() % std::max<int64_t>(batch_size, int64_t{1}) == 0,
      "expand_csr_rows_to_coo: col_indices elements must be divisible by batch count");
  const int64_t nnz_per_batch =
      col_indices.numel() / std::max<int64_t>(batch_size, int64_t{1});

  const int64_t batch_ndim = static_cast<int64_t>(batch_shape.size());
  const int64_t expected_rows = batch_ndim + 2;
  const int64_t total_nnz = col_indices.numel();

  TORCH_CHECK(
      coo_indices.dim() == 2 && coo_indices.size(0) == expected_rows &&
          coo_indices.size(1) == total_nnz,
      "expand_csr_rows_to_coo: output must have shape [",
      expected_rows,
      ", ",
      total_nnz,
      "]");

  Tensor crow_flat = crow_indices.reshape({batch_size, rows_plus_one}).contiguous();
  Tensor starts = crow_flat.slice(/*dim=*/1, /*start=*/0, /*end=*/rows_plus_one - 1);

  auto options_long = crow_indices.options().dtype(kLong);
  Tensor indicator = at::zeros({batch_size, nnz_per_batch}, options_long);
  if (rows_per_batch > 0 && nnz_per_batch > 0) {
    indicator.scatter_(1, starts, 1);
  }

  Tensor rows_flat = at::cumsum(indicator, 1);
  rows_flat.add_(-1);
  rows_flat = rows_flat.reshape({total_nnz});
  Tensor cols_flat = col_indices.reshape({total_nnz}).contiguous();

  std::array<int64_t, 2> expand_sizes{{batch_size, nnz_per_batch}};
  Tensor linear_matrix = at::arange(batch_size, options_long).unsqueeze(1).expand(expand_sizes);
  Tensor linear_flat = linear_matrix.reshape({total_nnz});

  std::vector<int64_t> strides(batch_ndim);
  int64_t stride_acc = 1;
  for (int64_t i = batch_ndim - 1; i >= 0; --i) {
    strides[i] = stride_acc;
    stride_acc *= batch_shape[i];
  }

  for (int64_t dim_idx = 0; dim_idx < batch_ndim; ++dim_idx) {
    int64_t size = batch_shape[dim_idx];
    if (size == 1) {
      coo_indices.select(0, dim_idx).zero_();
      continue;
    }
    int64_t stride = strides[dim_idx];
    Tensor coord = at::floor_divide(linear_flat, stride);
    coord = at::remainder(coord, size);
    coo_indices.select(0, dim_idx).copy_(coord.to(coo_indices.scalar_type()));
  }

  auto assign_row = [&](int64_t idx, const Tensor& src) {
    Tensor tmp = src.scalar_type() == coo_indices.scalar_type()
        ? src
        : src.to(coo_indices.scalar_type());
    coo_indices.select(0, idx).copy_(tmp);
  };

  if (transpose) {
    assign_row(batch_ndim, cols_flat);
    assign_row(batch_ndim + 1, rows_flat);
  } else {
    assign_row(batch_ndim, rows_flat);
    assign_row(batch_ndim + 1, cols_flat);
  }
}

static Tensor expand_csr_rows_to_coo(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    int64_t rows_per_batch,
    bool out_int32,
    bool transpose) {
  auto batch_shape = crow_indices.sizes().slice(0, crow_indices.dim() - 1);
  const int64_t batch_dim = std::accumulate(
      batch_shape.begin(), batch_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  const int64_t total_nnz = col_indices.numel();
  const int64_t nnz_per_batch =
      batch_dim > 0 ? total_nnz / std::max<int64_t>(batch_dim, int64_t{1}) : 0;
  const int64_t batch_ndim = static_cast<int64_t>(batch_shape.size());
  const int64_t expected_rows = batch_ndim + 2;
  auto options = crow_indices.options().dtype(out_int32 ? kInt : kLong);
  Tensor coo_indices = at::empty({expected_rows, total_nnz}, options);
  if (total_nnz == 0) {
    coo_indices.zero_();
    return coo_indices;
  }
  expand_csr_rows_to_coo_out(
      crow_indices,
      col_indices,
      rows_per_batch,
      out_int32,
      transpose,
      coo_indices);
  return coo_indices;
}

} // namespace csr
} // namespace mps

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Coalesce_metallib.h>
#endif

static Tensor compute_output_positions(const Tensor& is_unique) {

  int64_t nnz = is_unique.size(0);
  if (nnz == 0) {
    return at::empty({0}, TensorOptions().device(kMPS).dtype(kInt));
  }

  Tensor positions = at::empty({nnz}, TensorOptions().device(kMPS).dtype(kInt));

  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc("compute_output_positions_kernel");
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];

      mtl_setArgs(encoder, is_unique, positions);
      mtl_dispatch1DJob(encoder, pipeline, nnz);
    }
  });

  return positions;
}

static Tensor compute_output_positions_parallel(const Tensor& is_unique) {

  int64_t nnz = is_unique.size(0);
  if (nnz == 0) {
    return at::empty({0}, TensorOptions().device(kMPS).dtype(kInt));
  }

  // for small arrays, use simple kernel
  // speed of the naive kernel drops off after 4096 nnz elements
  if (nnz <= 4096) {
    return compute_output_positions(is_unique);
  }
  auto stream = getCurrentMPSStream();
  Tensor positions = is_unique.to(kInt);
  // Kogge-Stone parallel prefix sum
  Tensor positions_cloned = positions.clone();

  for (int64_t stride = 1; stride < nnz; stride *= 2) {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto pipeline = lib.getPipelineStateForFunc("kogge_stone_step");
        auto encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipeline];

        mtl_setArgs(encoder, positions, positions_cloned, stride);
        mtl_dispatch1DJob(encoder, pipeline, nnz);
      }
    });
    std::swap(positions, positions_cloned);
  }

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc("shift_right_kernel");
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];

      mtl_setArgs(encoder, positions, positions_cloned);
      mtl_dispatch1DJob(encoder, pipeline, nnz);
    }
  });

  return positions_cloned;
}

static std::pair<Tensor, int32_t> mark_unique_and_count(const Tensor& flat_indices) {

  int64_t nnz = flat_indices.size(0);
  if (nnz == 0) {
    return {at::empty({0}, flat_indices.options().dtype(kBool)), 0};
  }

  Tensor is_unique = at::empty({nnz}, flat_indices.options().dtype(kBool));
  Tensor count_result = at::zeros({1}, flat_indices.options().dtype(kInt));

  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc("mark_unique_positions_and_count_kernel");
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];

      mtl_setArgs(encoder, flat_indices, is_unique, count_result);
      mtl_dispatch1DJob(encoder, pipeline, nnz);
    }
  });

  int32_t num_unique = count_result.item<int32_t>();

  return {is_unique, num_unique};
}

SparseTensor _coalesce_sparse_mps(const SparseTensor& self) {
  int64_t nnz = self._nnz();
  TORCH_INTERNAL_ASSERT(!self.is_coalesced());
  if (nnz < 2) {
    SparseTensor dst = self.clone();
    dst._coalesced_(true);
    return dst;
  }

  Tensor indices = self._indices();
  Tensor values = self._values();

  Tensor flat_indices = flatten_indices(indices, self.sizes());
  Tensor sorted_order = flat_indices.argsort();
  Tensor flat_indices_sorted = flat_indices.index({sorted_order});
  values = values.index({sorted_order});
  indices = indices.index_select(1, sorted_order);

  auto unique_info = mark_unique_and_count(flat_indices_sorted);
  Tensor is_unique = unique_info.first;
  int32_t newNnz = unique_info.second;

  Tensor output_positions = compute_output_positions_parallel(is_unique);

  Tensor out_indices = at::empty({indices.size(0), newNnz}, indices.options());
  auto outValuesSize = values.sizes().vec();
  outValuesSize[0] = newNnz;
  Tensor out_values = at::zeros(outValuesSize, values.options());

  Tensor is_unique_local = is_unique;
  int64_t sparse_dim = indices.size(0);

  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc("coalesce_with_positions_kernel_" + scalarToMetalTypeString(values));
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];

      const uint32_t numThreads = static_cast<uint32_t>(nnz);
      const uint32_t valueSize = static_cast<uint32_t>(values.numel() / nnz);
      mtl_setArgs(encoder,
                  flat_indices_sorted,
                  indices,
                  values,
                  is_unique_local,
                  output_positions,
                  out_indices,
                  out_values,
                  numThreads,
                  valueSize,
                  sparse_dim,
                  newNnz);
      mtl_dispatch1DJob(encoder, pipeline, nnz);
    }
  });

  SparseTensor result = _sparse_coo_tensor_unsafe_symint(out_indices, out_values, self.sym_sizes())._coalesced_(true);
  return result;
}

TORCH_IMPL_FUNC(_convert_indices_from_coo_to_csr_structured_mps)
(const Tensor& input,
 const int64_t size,
 const bool out_int32,
 const Tensor& result) {
  TORCH_CHECK(
      input.is_mps() && result.is_mps(),
      "_convert_indices_from_coo_to_csr: expected MPS tensors");
  TORCH_CHECK(
      result.scalar_type() == (out_int32 ? kInt : kLong),
      "_convert_indices_from_coo_to_csr: output dtype mismatch");

  TORCH_CHECK(size >= 0, "_convert_indices_from_coo_to_csr: size must be non-negative");

  result.zero_();

  const int64_t nnz = input.numel();
  if (nnz == 0 || size == 0) {
    return;
  }

  Tensor rows = input.contiguous();
  if (rows.scalar_type() != kLong) {
    rows = rows.to(kLong);
  }

  auto options = rows.options().dtype(kLong);
  Tensor batch_ptr = at::zeros({2}, options);
  batch_ptr.narrow(0, 1, 1).fill_(nnz);

  Tensor row_ptr_long = out_int32 ? at::empty(result.sizes(), options) : result;

  mps::csr::build_row_ptr_per_batch_mps_out(
      rows,
      batch_ptr,
      /*batch_count=*/1,
      /*rows_per_batch=*/size,
      row_ptr_long);

  if (out_int32) {
    result.copy_(row_ptr_long.to(kInt));
  }
}

TORCH_IMPL_FUNC(_convert_indices_from_csr_to_coo_structured_mps)
(const Tensor& crow_indices,
 const Tensor& col_indices,
 const bool out_int32,
 const bool transpose,
 const Tensor& result) {
  TORCH_CHECK(
      crow_indices.is_mps() && col_indices.is_mps() && result.is_mps(),
      "_convert_indices_from_csr_to_coo: expected MPS tensors");

  TORCH_CHECK(
      result.scalar_type() == (out_int32 ? kInt : kLong),
      "_convert_indices_from_csr_to_coo: output dtype mismatch");

  if (result.numel() == 0) {
    result.zero_();
    return;
  }

  int64_t rows_per_batch = crow_indices.size(-1) - 1;
  TORCH_CHECK(
      rows_per_batch >= 0,
      "_convert_indices_from_csr_to_coo: invalid crow_indices shape");

  mps::csr::expand_csr_rows_to_coo_out(
      crow_indices,
      col_indices,
      rows_per_batch,
      out_int32,
      transpose,
      result);
}

void _validate_compressed_sparse_indices_mps(
    const bool is_crow,
    const Tensor& cidx,
    const Tensor& idx,
    const int64_t cdim,
    const int64_t dim,
    const int64_t nnz) {
  auto cidx_cpu = cidx.cpu();
  auto idx_cpu = idx.cpu();
  at::_validate_compressed_sparse_indices(
      is_crow,
      cidx_cpu,
      idx_cpu,
      cdim,
      dim,
      nnz);
}

} // namespace at::native