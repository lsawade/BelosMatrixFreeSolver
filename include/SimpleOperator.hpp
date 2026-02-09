#pragma once

// ---------------------------------------------------------------------
// SimpleOperator – abstract base for a matrix-free operator.
//
// Subclass this and implement apply() with your spectral element
// action.  Belos calls this through OperatorTraits::Apply().
// ---------------------------------------------------------------------

#include "SimpleMultiVec.hpp"

class SimpleOperator {
public:
  virtual ~SimpleOperator() = default;

  // Compute Y = A * X.
  // X and Y must have the same length and number of vectors.
  virtual void apply(const SimpleMultiVec& X, SimpleMultiVec& Y) const = 0;
};

// =====================================================================
// Laplacian1D – trivial test operator: (-1, 2, -1) stencil.
//
// This exists only to verify the Belos adapter works.
// Replace it with your spectral element operator.
// =====================================================================
class Laplacian1D : public SimpleOperator {
  int n_;

public:
  explicit Laplacian1D(int n) : n_(n) {}
  int n() const { return n_; }

  void apply(const SimpleMultiVec& X, SimpleMultiVec& Y) const override {
    for (int v = 0; v < X.numVecs(); ++v) {
      for (int i = 0; i < n_; ++i) {
        double val = 2.0 * X(i, v);
        if (i > 0)      val -= X(i - 1, v);
        if (i < n_ - 1) val -= X(i + 1, v);
        Y(i, v) = val;
      }
    }
  }
};

// =====================================================================
// SpectralElementOperator2D – skeleton for 2D SE gather/scatter.
//
// This shows the pattern your code would follow.  Fill in the
// "local action" section with your optimized element kernel.
//
// Data layout:
//   u(nglob)                          – global solution
//   mapping(nspec, ngll, ngll) -> iglob  – connectivity
//
// Apply pattern (for each element):
//   1. Gather:  uloc(iz,ix) = u( mapping(ispec, iz, ix) )
//   2. Local action: floc = K_local * uloc   (your SE kernel)
//   3. Scatter: f( mapping(ispec, iz, ix) ) += floc(iz,ix)
// =====================================================================
class SpectralElementOperator2D : public SimpleOperator {
  int nglob_;
  int nspec_;
  int ngll_;   // GLL points per direction

  // mapping_[ispec * ngll*ngll + iz * ngll + ix] -> iglob
  std::vector<int> mapping_;

public:
  SpectralElementOperator2D(int nglob, int nspec, int ngll,
                            const std::vector<int>& mapping)
      : nglob_(nglob), nspec_(nspec), ngll_(ngll), mapping_(mapping) {}

  int nglob() const { return nglob_; }

  void apply(const SimpleMultiVec& X, SimpleMultiVec& Y) const override {
    const int nv    = X.numVecs();
    const int ngll2 = ngll_ * ngll_;

    // Zero output
    for (int v = 0; v < nv; ++v)
      for (int i = 0; i < nglob_; ++i)
        Y(i, v) = 0.0;

    // Element loop
    for (int v = 0; v < nv; ++v) {
      for (int ispec = 0; ispec < nspec_; ++ispec) {
        const int base = ispec * ngll2;

        // ----- Gather -----
        std::vector<double> uloc(ngll2);
        for (int iz = 0; iz < ngll_; ++iz)
          for (int ix = 0; ix < ngll_; ++ix)
            uloc[iz * ngll_ + ix] =
                X(mapping_[base + iz * ngll_ + ix], v);

        // ----- Local element action -----
        // YOUR OPTIMIZED KERNEL GOES HERE.
        //
        // For a 2D spectral-element Laplacian this would be:
        //   floc = (Dx^T W Jac Dx + Dz^T W Jac Dz) uloc
        // using tensor-product structure for efficiency.
        //
        // Placeholder: identity (floc = uloc), so the assembled
        // operator is a lumped "mass-like" matrix.
        std::vector<double> floc(uloc);

        // ----- Scatter / accumulate -----
        for (int iz = 0; iz < ngll_; ++iz)
          for (int ix = 0; ix < ngll_; ++ix)
            Y(mapping_[base + iz * ngll_ + ix], v) +=
                floc[iz * ngll_ + ix];
      }
    }
  }
};
