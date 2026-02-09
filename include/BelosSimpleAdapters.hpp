#pragma once

// ---------------------------------------------------------------------
// Belos adapter traits for SimpleMultiVec / SimpleOperator.
//
// These two specializations are all Belos needs to drive CG, GMRES,
// etc. with your own data structures.  No Tpetra anywhere.
//
// Belos::MultiVecTraits<double, SimpleMultiVec>
//   – tells Belos how to clone, dot, norm, update, etc.
//
// Belos::OperatorTraits<double, SimpleMultiVec, SimpleOperator>
//   – tells Belos how to apply your operator.
// ---------------------------------------------------------------------

#include "SimpleMultiVec.hpp"
#include "SimpleOperator.hpp"

#include <BelosConfigDefs.hpp>
#include <BelosMultiVecTraits.hpp>
#include <BelosOperatorTraits.hpp>
#include <BelosTypes.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_RCP.hpp>

#include <cmath>
#include <random>
#include <iostream>

namespace Belos {

// =====================================================================
//  MultiVecTraits<double, SimpleMultiVec>
// =====================================================================
template <>
class MultiVecTraits<double, SimpleMultiVec> {
public:
  using MV = SimpleMultiVec;

  // ---------- creation ------------------------------------------------

  // New MV, same length, different number of columns, uninitialised.
  static Teuchos::RCP<MV> Clone(const MV& mv, const int numvecs) {
    return Teuchos::rcp(new MV(mv.length(), numvecs));
  }

  // Deep copy, all columns.
  static Teuchos::RCP<MV> CloneCopy(const MV& mv) {
    auto r = Teuchos::rcp(new MV(mv.length(), mv.numVecs()));
    for (int j = 0; j < mv.numVecs(); ++j)
      for (int i = 0; i < mv.length(); ++i)
        (*r)(i, j) = mv(i, j);
    return r;
  }

  // Deep copy, selected columns.
  static Teuchos::RCP<MV> CloneCopy(const MV& mv,
                                    const std::vector<int>& index) {
    const int nv = static_cast<int>(index.size());
    auto r = Teuchos::rcp(new MV(mv.length(), nv));
    for (int j = 0; j < nv; ++j)
      for (int i = 0; i < mv.length(); ++i)
        (*r)(i, j) = mv(i, index[j]);
    return r;
  }

  // Mutable view of selected columns (shared data, no copy).
  static Teuchos::RCP<MV> CloneViewNonConst(MV& mv,
                                            const std::vector<int>& index) {
    std::vector<MV::col_ptr> cols;
    cols.reserve(index.size());
    for (int j : index)
      cols.push_back(mv.getCol(j));
    return Teuchos::rcp(new MV(mv.length(), std::move(cols)));
  }

  // Const view of selected columns.
  static Teuchos::RCP<const MV> CloneView(const MV& mv,
                                          const std::vector<int>& index) {
    std::vector<MV::col_ptr> cols;
    cols.reserve(index.size());
    for (int j : index)
      cols.push_back(mv.getCol(j));
    return Teuchos::rcp(
        static_cast<const MV*>(new MV(mv.length(), std::move(cols))));
  }

  // ---------- attributes ----------------------------------------------

  static ptrdiff_t GetGlobalLength(const MV& mv) {
    return static_cast<ptrdiff_t>(mv.length());
  }

  static int GetNumberVecs(const MV& mv) { return mv.numVecs(); }

  static bool HasConstantStride(const MV& /*mv*/) { return false; }

  // ---------- update --------------------------------------------------

  // mv = alpha * A * B + beta * mv
  //   A  : MV  (m x p)
  //   B  : dense (p x q)
  //   mv : MV  (m x q)
  static void MvTimesMatAddMv(const double alpha, const MV& A,
                              const Teuchos::SerialDenseMatrix<int, double>& B,
                              const double beta, MV& mv) {
    const int m = A.length();
    const int p = A.numVecs();
    const int q = mv.numVecs();
    for (int j = 0; j < q; ++j) {
      if (beta == 0.0)
        for (int i = 0; i < m; ++i) mv(i, j) = 0.0;
      else
        for (int i = 0; i < m; ++i) mv(i, j) *= beta;
      for (int k = 0; k < p; ++k) {
        const double c = alpha * B(k, j);
        for (int i = 0; i < m; ++i)
          mv(i, j) += c * A(i, k);
      }
    }
  }

  // mv = alpha * A + beta * B
  static void MvAddMv(const double alpha, const MV& A,
                      const double beta,  const MV& B, MV& mv) {
    const int m  = A.length();
    const int nv = A.numVecs();
    for (int j = 0; j < nv; ++j)
      for (int i = 0; i < m; ++i)
        mv(i, j) = alpha * A(i, j) + beta * B(i, j);
  }

  // mv *= alpha  (all columns)
  static void MvScale(MV& mv, const double alpha) {
    for (int j = 0; j < mv.numVecs(); ++j)
      for (int i = 0; i < mv.length(); ++i)
        mv(i, j) *= alpha;
  }

  // mv(:,j) *= alphas[j]
  static void MvScale(MV& mv, const std::vector<double>& alphas) {
    for (int j = 0; j < mv.numVecs(); ++j)
      for (int i = 0; i < mv.length(); ++i)
        mv(i, j) *= alphas[j];
  }

  // C = alpha * A^T * B   (dense output, p x q)
  static void MvTransMv(const double alpha, const MV& A, const MV& B,
                        Teuchos::SerialDenseMatrix<int, double>& C) {
    const int m = A.length();
    const int p = A.numVecs();
    const int q = B.numVecs();
    C.shape(p, q);
    for (int i = 0; i < p; ++i)
      for (int j = 0; j < q; ++j) {
        double dot = 0.0;
        for (int k = 0; k < m; ++k)
          dot += A(k, i) * B(k, j);
        C(i, j) = alpha * dot;
      }
  }

  // dots[j] = A(:,j) . B(:,j)
  static void MvDot(const MV& A, const MV& B, std::vector<double>& dots) {
    const int nv = A.numVecs();
    dots.resize(nv);
    for (int j = 0; j < nv; ++j) {
      double d = 0.0;
      for (int i = 0; i < A.length(); ++i)
        d += A(i, j) * B(i, j);
      dots[j] = d;
    }
  }

  // ---------- norms ---------------------------------------------------

  static void MvNorm(const MV& mv, std::vector<double>& normvec,
                     NormType type = TwoNorm) {
    const int nv = mv.numVecs();
    normvec.resize(nv);
    for (int j = 0; j < nv; ++j) {
      double val = 0.0;
      for (int i = 0; i < mv.length(); ++i) {
        const double v = mv(i, j);
        if (type == TwoNorm)
          val += v * v;
        else if (type == OneNorm)
          val += std::abs(v);
        else  // InfNorm
          val = std::max(val, std::abs(v));
      }
      normvec[j] = (type == TwoNorm) ? std::sqrt(val) : val;
    }
  }

  // ---------- copy / init ---------------------------------------------

  // mv(:, index[j]) = A(:, j)
  static void SetBlock(const MV& A, const std::vector<int>& index, MV& mv) {
    for (int j = 0; j < static_cast<int>(index.size()); ++j)
      for (int i = 0; i < mv.length(); ++i)
        mv(i, index[j]) = A(i, j);
  }

  // Deep copy: mv = A  (same dimensions assumed)
  static void Assign(const MV& A, MV& mv) {
    for (int j = 0; j < mv.numVecs(); ++j)
      for (int i = 0; i < mv.length(); ++i)
        mv(i, j) = A(i, j);
  }

  static void MvRandom(MV& mv) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int j = 0; j < mv.numVecs(); ++j)
      for (int i = 0; i < mv.length(); ++i)
        mv(i, j) = dist(gen);
  }

  static void MvInit(MV& mv, const double alpha = 0.0) {
    for (int j = 0; j < mv.numVecs(); ++j)
      for (int i = 0; i < mv.length(); ++i)
        mv(i, j) = alpha;
  }

  static void MvPrint(const MV& mv, std::ostream& os) {
    for (int i = 0; i < mv.length(); ++i) {
      for (int j = 0; j < mv.numVecs(); ++j)
        os << mv(i, j) << "\t";
      os << "\n";
    }
  }
};

// =====================================================================
//  OperatorTraits<double, SimpleMultiVec, SimpleOperator>
// =====================================================================
template <>
class OperatorTraits<double, SimpleMultiVec, SimpleOperator> {
public:
  static void Apply(const SimpleOperator& Op,
                    const SimpleMultiVec& x, SimpleMultiVec& y,
                    ETrans /*trans*/ = NOTRANS) {
    Op.apply(x, y);
  }

  static bool HasApplyTranspose(const SimpleOperator& /*Op*/) {
    return false;
  }
};

}  // namespace Belos
