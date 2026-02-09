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
