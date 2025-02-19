#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <deque>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Maps TID/BID to its dimension. It is by default blockDim/gridDim,
//! but if use of a ParallelType is mapped to a unique constant
//! extent, the constant value is used instead since presumably it's
//! more efficient.
class TORCH_CUDA_CU_API ParallelDimensionMap {
 public:
  void build(Fusion* fusion);

  //! Returns the dimension of a ParallelType. nullptr is returned if
  //! a ParallelType is unused.
  kir::Val* get(ParallelType pt) const;

  //! True if the dimension of a ParallelType is known to be exact
  bool isExact(ParallelType pt) const;

  std::string toString() const;

  //! Symbolically analyze if two extent vals are equal
  static bool equalDim(kir::Val* dim1, kir::Val* dim2);

 private:
  //! Register the extent of an IterDomain if its constant
  void registerConstantExtent(IterDomain* id);

  void handleParallelDomain(IterDomain* id);

  void populateDimensionMapWithSingleCASet(
      ParallelType pt,
      const std::unordered_set<IterDomain*>& dom_set);

  void populateDimensionMapWithMultipleCASet(
      ParallelType pt,
      const std::unordered_set<IterDomain*>& dom_set);

  static IterDomain* getCAMappedConcreteDomain(IterDomain* id);

 private:
  //! Maps from parallel types to dimensions, which are constant if
  //! a unique value is found.
  std::unordered_map<ParallelType, kir::Val*, TypeHash> dim_map_;
  //! Set of parallel types whose dimensions are identified to be
  //! exactly the same as extents of mapped domains.
  std::unordered_set<ParallelType, TypeHash> exact_types_;

  // Below are temporary maps to build the ParallelType-to-dimension
  // map. Only used during build().

  //! Map from a parallel type to a set of concrete domains where the
  //! parallel type is used.
  std::unordered_map<ParallelType, std::unordered_set<IterDomain*>, TypeHash>
      concrete_dom_map_;
  //! Keep track of constant extents found for a CA domain set
  //! represented by the concrete domain.
  std::unordered_map<IterDomain*, std::unordered_set<int64_t>>
      constant_extent_map_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
