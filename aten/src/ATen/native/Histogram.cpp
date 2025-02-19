#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>

#include <ATen/native/Histogram.h>
#include <ATen/native/Resize.h>

#include <numeric>
#include <tuple>
#include <vector>
#include <functional>
#include <c10/util/ArrayRef.h>
#include <c10/core/ScalarType.h>
#include <c10/core/DefaultDtype.h>

/* Implements a numpy-like histogramdd function running on cpu
 * https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html
 *
 * See the docstr for torch.histogramdd in torch/functional.py for further explanation.
 *
 * - torch.histogramdd(input, bins, range=None, weight=None, density=False)
 *   input     - tensor with shape (M, N). input is interpreted as M coordinates in N-dimensional space.
 *               If a tensor with more than 2 dimensions is passed, all but the last dimension will be flattened.
 *   bins      - int[] of length N or tensor list of length N. If int[], defines the number of equal-width bins
 *               in each dimension. If tensor list, defines the sequences of bin edges, including rightmost edges,
 *               for each dimension.
 *   range     - float[] of length 2 * N, optional. If specified, defines the leftmost and rightmost bin edges
 *               for each dimension.
 *   weight    - tensor, optional. If provided, weight should have the same shape as input excluding its last dimension.
 *               Each N-dimensional value in input contributes its associated weight towards its bin's result.
 *               If weight is not specified, each value has weight 1 by default.
 *   density   - bool, optional. If false (default), the result will contain the total count (weight) in each bin.
 *               If True, each count (weight) is divided by the total count (total weight), then divided by the
 *               volume of its associated bin.
 *
 * Returns:
 *   hist      - N-dimensional tensor containing the values of the histogram.
 *   bin_edges - tensor list of length N containing the edges of the histogram bins in each dimension.
 *               Bins include their left edge and exclude their right edge, with the exception of the
 *               rightmost bin in each dimension which includes both of its edges.
 *
 * Restrictions are defined in histogram_check_inputs() and in select_outer_bin_edges().
 */

namespace at { namespace native {

DEFINE_DISPATCH(histogramdd_stub);
DEFINE_DISPATCH(histogramdd_linear_stub);

namespace {

/* Checks properties of input tensors input, bins, and weight.
 */
void histogramdd_check_inputs(const Tensor& input, const TensorList& bins, const c10::optional<Tensor>& weight) {
    TORCH_CHECK(input.dim() >= 2, "torch.histogramdd: input tensor should have at least 2 dimensions, but got ",
                input.dim());

    const int64_t N = input.size(-1);

    TORCH_CHECK(bins.size() == N, "torch.histogramdd: expected ", N, " sequences of bin edges for a ", N,
                "-dimensional histogram but got ", bins.size());

    auto input_dtype = input.dtype();
    for (int64_t dim = 0; dim < N; dim++) {
        const Tensor& dim_bins = bins[dim];

        auto bins_dtype = dim_bins.dtype();
        TORCH_CHECK(input_dtype == bins_dtype, "torch.histogramdd: input tensor and bins tensors should",
                " have the same dtype, but got input with dtype ", input_dtype,
                " and bins for dimension ", dim, " with dtype ", bins_dtype);

        const int64_t dim_bins_dim = dim_bins.dim();
        TORCH_CHECK(dim_bins_dim == 1, "torch.histogramdd: bins tensor should have one dimension,",
                " but got ", dim_bins_dim, " dimensions in the bins tensor for dimension ", dim);

        const int64_t numel = dim_bins.numel();
        TORCH_CHECK(numel > 0, "torch.histogramdd: bins tensor should have at least 1 element,",
                " but got ", numel, " elements in the bins tensor for dimension ", dim);
    }

    if (weight.has_value()) {
        TORCH_CHECK(input.dtype() == weight.value().dtype(), "torch.histogramdd: if weight tensor is provided,"
                " input tensor and weight tensor should have the same dtype, but got input(", input.dtype(), ")",
                ", and weight(", weight.value().dtype(), ")");

        /* If a weight tensor is provided, we expect its shape to match that of
         * the input tensor excluding its innermost dimension N.
         */
        auto input_sizes = input.sizes().vec();
        input_sizes.pop_back();

        auto weight_sizes = weight.value().sizes().vec();
        if (weight_sizes.empty()) {
            // correctly handle scalars
            weight_sizes = {1};
        }

        TORCH_CHECK(input_sizes == weight_sizes, "torch.histogramdd: if weight tensor is provided it should have"
                " the same shape as the input tensor excluding its innermost dimension, but got input with shape ",
                input.sizes(), " and weight with shape ", weight.value().sizes());
    }
}

/* Checks properties of output tensors hist and bin_edges, then resizes them.
 */
void histogramdd_prepare_out(const Tensor& input, const std::vector<int64_t>& bin_ct,
        const Tensor& hist, const TensorList& bin_edges) {
    const int64_t N = input.size(-1);

    TORCH_INTERNAL_ASSERT((int64_t)bin_ct.size() == N);
    TORCH_INTERNAL_ASSERT((int64_t)bin_edges.size() == N);

    TORCH_CHECK(input.dtype() == hist.dtype(), "torch.histogram: input tensor and hist tensor should",
            " have the same dtype, but got input ", input.dtype(), " and hist ", hist.dtype());

    for (int64_t dim = 0; dim < N; dim++) {
        TORCH_CHECK(input.dtype() == bin_edges[dim].dtype(), "torch.histogram: input tensor and bin_edges tensor should",
                " have the same dtype, but got input ", input.dtype(), " and bin_edges ", bin_edges[dim].dtype(),
                " for dimension ", dim);

        TORCH_CHECK(bin_ct[dim] > 0,
                "torch.histogram(): bins must be > 0, but got ", bin_ct[dim], " for dimension ", dim);

        at::native::resize_output(bin_edges[dim], bin_ct[dim] + 1);
    }

    at::native::resize_output(hist, bin_ct);
}

void histogramdd_prepare_out(const Tensor& input, TensorList bins,
        const Tensor& hist, const TensorList& bin_edges) {
    std::vector<int64_t> bin_ct(bins.size());
    std::transform(bins.begin(), bins.end(), bin_ct.begin(), [](Tensor t) { return t.numel() - 1; });
    histogramdd_prepare_out(input, bin_ct, hist, bin_edges);
}

template<typename scalar_t>
void infer_bin_edges_from_input(const Tensor& input, const int64_t N,
        std::vector<double> &leftmost_edges, std::vector<double> &rightmost_edges) {
    // Calls aminmax on input with dim=0, reducing all but the innermost dimension of input.
    Tensor min, max;
    std::tie(min, max) = aminmax(input, 0);

    TORCH_INTERNAL_ASSERT(min.is_contiguous() && max.is_contiguous());

    const scalar_t *min_data = min.data_ptr<scalar_t>();
    std::copy(min_data, min_data + N, leftmost_edges.begin());

    const scalar_t *max_data = max.data_ptr<scalar_t>();
    std::copy(max_data, max_data + N, rightmost_edges.begin());
}

/* Determines the outermost bin edges. For simplicity when calling into aminmax,
 * assumes that input has already been reshaped to (M, N).
 */
std::pair<std::vector<double>, std::vector<double>>
select_outer_bin_edges(const Tensor& input, c10::optional<c10::ArrayRef<double>> range) {
    TORCH_INTERNAL_ASSERT(input.dim() == 2, "expected input to have shape (M, N)");
    const int64_t N = input.size(-1);

    // Default ranges for empty input matching numpy.histogram's default
    std::vector<double> leftmost_edges(N, 0.);
    std::vector<double> rightmost_edges(N, 1.);

    if (range.has_value()) {
        // range is specified
        TORCH_CHECK((int64_t)range.value().size() == 2 * N, "torch.histogramdd: for a ", N, "-dimensional histogram",
                " range should have ", 2 * N, " elements, but got ", range.value().size());

        for (int64_t dim = 0; dim < N; dim++) {
            leftmost_edges[dim] = range.value()[2 * dim];
            rightmost_edges[dim] = range.value()[2 * dim + 1];
        }
    } else if (input.numel() > 0) {
        // non-empty input
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "histogramdd", [&]() {
            infer_bin_edges_from_input<scalar_t>(input, N, leftmost_edges, rightmost_edges);
        });
    }

    for (int64_t dim = 0; dim < N; dim++) {
        double leftmost_edge = leftmost_edges[dim];
        double rightmost_edge = rightmost_edges[dim];

        TORCH_CHECK(std::isfinite(leftmost_edge) && std::isfinite(rightmost_edge),
                "torch.histogramdd: dimension ", dim, "'s range [",
                leftmost_edge, ", ", rightmost_edge, "] is not finite");

        TORCH_CHECK(leftmost_edge <= rightmost_edge, "torch.histogramdd: min should not exceed max, but got",
                " min ", leftmost_edge, " max ", rightmost_edge, " for dimension ", dim);

        // Expand empty range to match numpy behavior and avoid division by 0 in normalization
        if (leftmost_edge == rightmost_edge) {
            leftmost_edges[dim] -= 0.5;
            rightmost_edges[dim] += 0.5;
        }
    }

    return std::make_pair(leftmost_edges, rightmost_edges);
}

/* histc's version of the logic for outermost bin edges.
 */
std::pair<double, double> histc_select_outer_bin_edges(const Tensor& input,
        const Scalar& min, const Scalar& max) {
    double leftmost_edge = min.to<double>();
    double rightmost_edge = max.to<double>();

    if (leftmost_edge == rightmost_edge) {
        auto extrema = _aminmax(input);
        leftmost_edge = std::get<0>(extrema).item<double>();
        rightmost_edge = std::get<1>(extrema).item<double>();
    }

    if (leftmost_edge == rightmost_edge) {
        leftmost_edge -= 1;
        rightmost_edge += 1;
    }

    TORCH_CHECK(!(std::isinf(leftmost_edge) || std::isinf(rightmost_edge) ||
            std::isnan(leftmost_edge) || std::isnan(rightmost_edge)),
            "torch.histc: range of [", leftmost_edge, ", ", rightmost_edge, "] is not finite");

    TORCH_CHECK(leftmost_edge < rightmost_edge, "torch.histc: max must be larger than min");

    return std::make_pair(leftmost_edge, rightmost_edge);
}

} // namespace

std::vector<Tensor> allocate_bin_edges_tensors(const Tensor& self) {
    TORCH_CHECK(self.dim() >= 2, "torch.histogramdd: input tensor should have at least 2 dimensions");
    const int64_t N = self.size(-1);
    std::vector<Tensor> bin_edges_out(N);
    for (int64_t dim = 0; dim < N; dim++) {
        bin_edges_out[dim] = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    }
    return bin_edges_out;
}

/* Versions of histogramdd in which bins is a Tensor[] defining the sequences of bin edges.
 */
Tensor& histogramdd_out_cpu(const Tensor& self, TensorList bins,
        const c10::optional<Tensor>& weight, bool density,
        Tensor& hist, TensorList& bin_edges) {
    histogramdd_check_inputs(self, bins, weight);
    histogramdd_prepare_out(self, bins, hist, bin_edges);

    for (size_t dim = 0; dim < bins.size(); dim++) {
        bin_edges[dim].copy_(bins[dim]);
    }

    histogramdd_stub(self.device().type(), self, weight, density, hist, bin_edges);
    return hist;
}

Tensor histogramdd_cpu(const Tensor& self, TensorList bins,
        const c10::optional<Tensor>& weight, bool density) {
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    std::vector<Tensor> bin_edges_out = allocate_bin_edges_tensors(self);
    TensorList bin_edges_out_tl(bin_edges_out);

    histogramdd_out_cpu(self, bins, weight, density, hist, bin_edges_out_tl);
    return hist;
}

/* Versions of histogramdd in which bins is an int[]
 * defining the number of bins in each dimension.
 */
std::vector<Tensor>& histogramdd_bin_edges_out_cpu(const Tensor& self, IntArrayRef bin_ct,
        c10::optional<c10::ArrayRef<double>> range,
        const c10::optional<Tensor>& weight, bool density,
        std::vector<Tensor>& bin_edges_out) {
    TensorList bin_edges_out_tl(bin_edges_out);

    const int64_t N = self.size(-1);
    const int64_t M = std::accumulate(self.sizes().begin(), self.sizes().end() - 1,
            (int64_t)1, std::multiplies<int64_t>());
    Tensor reshaped_self = self.reshape({ M, N });

    auto outer_bin_edges = select_outer_bin_edges(reshaped_self, range);

    for (int64_t dim = 0; dim < N; dim++) {
        linspace_cpu_out(outer_bin_edges.first[dim], outer_bin_edges.second[dim],
                bin_ct[dim] + 1, bin_edges_out[dim]);
    }

    return bin_edges_out;
}

std::vector<Tensor> histogramdd_bin_edges_cpu(const Tensor& self, IntArrayRef bin_ct,
        c10::optional<c10::ArrayRef<double>> range,
        const c10::optional<Tensor>& weight, bool density) {
    std::vector<Tensor> bin_edges_out = allocate_bin_edges_tensors(self);
    return histogramdd_bin_edges_out_cpu(self, bin_ct, range, weight, density, bin_edges_out);
}

Tensor& histogramdd_out_cpu(const Tensor& self, IntArrayRef bin_ct,
        c10::optional<c10::ArrayRef<double>> range,
        const c10::optional<Tensor>& weight, bool density,
        Tensor& hist, TensorList& bin_edges) {
    std::vector<Tensor> bins = histogramdd_bin_edges_cpu(self, bin_ct, range, weight, density);

    histogramdd_check_inputs(self, bins, weight);
    histogramdd_prepare_out(self, bins, hist, bin_edges);

    for (size_t dim = 0; dim < bins.size(); dim++) {
        bin_edges[dim].copy_(bins[dim]);
    }

    histogramdd_linear_stub(self.device().type(), self, weight, density, hist, bin_edges, true);
    return hist;
}

Tensor histogramdd_cpu(const Tensor& self, IntArrayRef bin_ct,
        c10::optional<c10::ArrayRef<double>> range,
        const c10::optional<Tensor>& weight, bool density) {
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    std::vector<Tensor> bin_edges_out = allocate_bin_edges_tensors(self);
    TensorList bin_edges_out_tl(bin_edges_out);

    histogramdd_out_cpu(self, bin_ct, range, weight, density, hist, bin_edges_out_tl);
    return hist;
}

/* Versions of histogram in which bins is a Tensor defining the sequence of bin edges.
 */
std::tuple<Tensor&, Tensor&>
histogram_out_cpu(const Tensor& self, const Tensor& bins,
        const c10::optional<Tensor>& weight, bool density,
        Tensor& hist, Tensor& bin_edges) {
    Tensor reshaped_self = self.reshape({ self.numel(), 1 });
    c10::optional<Tensor> reshaped_weight = weight.has_value()
        ? weight.value().reshape({ weight.value().numel() }) : weight;
    TensorList bins_in = bins;
    TensorList bins_out = bin_edges;

    histogramdd_out_cpu(reshaped_self, bins_in, reshaped_weight, density, hist, bins_out);

    return std::forward_as_tuple(hist, bin_edges);
}

std::tuple<Tensor, Tensor>
histogram_cpu(const Tensor& self, const Tensor& bins,
        const c10::optional<Tensor>& weight, bool density) {
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    Tensor bin_edges = at::empty({0}, bins.options(), MemoryFormat::Contiguous);
    return histogram_out_cpu(self, bins, weight, density, hist, bin_edges);
}

/* Versions of histogram in which bins is an integer specifying the number of equal-width bins.
 */
std::tuple<Tensor&, Tensor&>
histogram_out_cpu(const Tensor& self, int64_t bin_ct, c10::optional<c10::ArrayRef<double>> range,
        const c10::optional<Tensor>& weight, bool density,
        Tensor& hist, Tensor& bin_edges) {
    Tensor reshaped_self = self.reshape({ self.numel(), 1 });
    c10::optional<Tensor> reshaped_weight = weight.has_value()
        ? weight.value().reshape({ weight.value().numel() }) : weight;
    TensorList bins_in = bin_edges;
    TensorList bins_out = bin_edges;

    histogramdd_prepare_out(reshaped_self, std::vector<int64_t>{bin_ct}, hist, bins_out);
    auto outer_bin_edges = select_outer_bin_edges(reshaped_self, range);
    linspace_cpu_out(outer_bin_edges.first[0], outer_bin_edges.second[0], bin_ct + 1, bin_edges);

    histogramdd_check_inputs(reshaped_self, bins_in, reshaped_weight);

    histogramdd_linear_stub(reshaped_self.device().type(), reshaped_self, reshaped_weight, density, hist, bin_edges, true);
    return std::forward_as_tuple(hist, bin_edges);
}

std::tuple<Tensor, Tensor>
histogram_cpu(const Tensor& self, int64_t bin_ct, c10::optional<c10::ArrayRef<double>> range,
        const c10::optional<Tensor>& weight, bool density) {
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    Tensor bin_edges_out = at::empty({0}, self.options());
    return histogram_out_cpu(self, bin_ct, range, weight, density, hist, bin_edges_out);
}

/* Narrowed interface for the legacy torch.histc function.
 */
Tensor& histogram_histc_cpu_out(const Tensor& self, int64_t bin_ct,
        const Scalar& min, const Scalar& max, Tensor& hist) {
    Tensor bin_edges = at::empty({0}, self.options());

    Tensor reshaped = self.reshape({ self.numel(), 1 });
    TensorList bins_in = bin_edges;
    TensorList bins_out = bin_edges;

    histogramdd_prepare_out(reshaped, std::vector<int64_t>{bin_ct}, hist, bins_out);

    auto outer_bin_edges = histc_select_outer_bin_edges(self, min, max);
    linspace_cpu_out(outer_bin_edges.first, outer_bin_edges.second, bin_ct + 1, bin_edges);

    histogramdd_check_inputs(reshaped, bins_in, {});

    histogramdd_linear_stub(reshaped.device().type(), reshaped,
            c10::optional<Tensor>(), false, hist, bin_edges, false);
    return hist;
}

Tensor histogram_histc_cpu(const Tensor& self, int64_t bin_ct,
        const Scalar& min, const Scalar& max) {
    Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
    return histogram_histc_cpu_out(self, bin_ct, min, max, hist);
}

}} // namespace at::native
