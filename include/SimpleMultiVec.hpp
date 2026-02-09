#pragma once

// ---------------------------------------------------------------------
// SimpleMultiVec – a lightweight multivector for use with Belos.
//
// Stores `numvecs` column vectors, each of length `length`.
// Columns are individually heap-allocated so that views (shared
// columns) are trivial: a view just copies the shared_ptrs.
//
// For your spectral element code with u(nglob, ndim):
//   length  = nglob  (for a scalar solve like Laplacian)
//         or  nglob * ndim  (if you flatten components)
//   numvecs = Belos block size (typically 1)
// ---------------------------------------------------------------------

#include <vector>
#include <memory>
#include <cassert>

class SimpleMultiVec {
public:
  using col_ptr = std::shared_ptr<std::vector<double>>;

private:
  int length_;
  std::vector<col_ptr> cols_;

public:
  SimpleMultiVec() : length_(0) {}

  // Owning constructor: allocates numvecs zero-filled columns.
  SimpleMultiVec(int length, int numvecs) : length_(length) {
    cols_.reserve(numvecs);
    for (int i = 0; i < numvecs; ++i)
      cols_.push_back(std::make_shared<std::vector<double>>(length, 0.0));
  }

  // View constructor: shares column data (no copy).
  SimpleMultiVec(int length, std::vector<col_ptr> cols)
      : length_(length), cols_(std::move(cols)) {}

  int length()  const { return length_; }
  int numVecs() const { return static_cast<int>(cols_.size()); }

  // Element access: (row i, column j)
  double& operator()(int i, int j)       { return (*cols_[j])[i]; }
  double  operator()(int i, int j) const { return (*cols_[j])[i]; }

  // Raw pointer to column data (for interfacing with your SE code)
  double*       colPtr(int j)       { return cols_[j]->data(); }
  const double* colPtr(int j) const { return cols_[j]->data(); }

  // Shared pointer to a column (used by view operations in the traits)
  col_ptr        getCol(int j)       { return cols_[j]; }
  const col_ptr& getCol(int j) const { return cols_[j]; }
};
