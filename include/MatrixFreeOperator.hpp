#pragma once

// ---------------------------------------------------------------------
// MatrixFreeOperator – a minimal Tpetra::Operator subclass that applies
// the 1-D Laplacian stencil (-1, 2, -1) *without* storing a matrix.
//
// This is the pattern you'd follow for any matrix-free operator:
//   1.  Inherit from Tpetra::Operator<SC,LO,GO,NO>
//   2.  Store the domain/range Maps (they are the same for a square op)
//   3.  Override getDomainMap(), getRangeMap(), apply()
//   4.  In apply(), implement  Y := alpha * Op(X) + beta * Y
//       (Op = A, A^T, or A^H depending on the mode argument)
//
// For your own operator, replace the stencil logic in apply() with
// whatever action you need.
// ---------------------------------------------------------------------

#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_Import.hpp>
#include <Teuchos_RCP.hpp>

template <class Scalar        = Tpetra::Operator<>::scalar_type,
          class LocalOrdinal  = Tpetra::Operator<>::local_ordinal_type,
          class GlobalOrdinal = Tpetra::Operator<>::global_ordinal_type,
          class Node          = Tpetra::Operator<>::node_type>
class MatrixFreeOperator
    : public Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> {
 public:
  // --- Tpetra typedefs (mirrors what CrsMatrix exposes) ---
  using scalar_type         = Scalar;
  using local_ordinal_type  = LocalOrdinal;
  using global_ordinal_type = GlobalOrdinal;
  using node_type           = Node;

  using map_type    = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
  using mv_type     = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using import_type = Tpetra::Import<LocalOrdinal, GlobalOrdinal, Node>;

  // ----- constructor ---------------------------------------------------
  // n : global number of rows/columns (square operator)
  MatrixFreeOperator(GlobalOrdinal n,
                     const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
  {
    const GlobalOrdinal indexBase = 0;

    // Non-overlapping (owned) map – Tpetra distributes rows evenly.
    opMap_ = Teuchos::rcp(new map_type(n, indexBase, comm));

    // Build an overlapping map that includes one ghost on each side so
    // the stencil can reach its neighbours.
    const int myRank   = comm->getRank();
    const int numProcs = comm->getSize();

    LocalOrdinal nLocal = opMap_->getLocalNumElements();
    if (myRank > 0)              ++nLocal;   // left ghost
    if (myRank < numProcs - 1)   ++nLocal;   // right ghost

    std::vector<GlobalOrdinal> indices;
    indices.reserve(nLocal);

    if (myRank > 0)
      indices.push_back(opMap_->getMinGlobalIndex() - 1);

    for (GlobalOrdinal i = opMap_->getMinGlobalIndex();
         i <= opMap_->getMaxGlobalIndex(); ++i)
      indices.push_back(i);

    if (myRank < numProcs - 1)
      indices.push_back(opMap_->getMaxGlobalIndex() + 1);

    const GlobalOrdinal numGlobal = n + 2 * (numProcs - 1);
    Teuchos::ArrayView<const GlobalOrdinal> elemList(indices);
    redistMap_ = Teuchos::rcp(new map_type(numGlobal, elemList, indexBase, comm));

    importer_ = Teuchos::rcp(new import_type(opMap_, redistMap_));
  }

  // ----- Tpetra::Operator interface ------------------------------------
  Teuchos::RCP<const map_type> getDomainMap() const override {
    return opMap_;
  }

  Teuchos::RCP<const map_type> getRangeMap() const override {
    return opMap_;
  }

  /// Compute Y := alpha * A * X + beta * Y  (matrix-free)
  void apply(const mv_type& X,
             mv_type& Y,
             Teuchos::ETransp mode = Teuchos::NO_TRANS,
             Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
             Scalar beta  = Teuchos::ScalarTraits<Scalar>::zero()) const override
  {
    using STS = Teuchos::ScalarTraits<Scalar>;

    const auto comm     = opMap_->getComm();
    const int myRank    = comm->getRank();
    const int numProcs  = comm->getSize();
    const size_t numVec = X.getNumVectors();
    const LocalOrdinal nRows =
        static_cast<LocalOrdinal>(X.getLocalLength());

    // Import X into the overlapping (ghosted) layout.
    mv_type ghostX(redistMap_, numVec);
    ghostX.doImport(X, *importer_, Tpetra::INSERT);

    // Scale Y by beta (handles the beta * Y_old part).
    if (beta == STS::zero())
      Y.putScalar(STS::zero());
    else if (beta != STS::one())
      Y.scale(beta);

    // Apply the stencil (-1, 2, -1) column-by-column.
    for (size_t c = 0; c < numVec; ++c) {
      auto col = ghostX.getData(c);

      // Offset into the ghost vector: rank 0 has no left ghost.
      const LocalOrdinal off = (myRank > 0) ? 0 : 1;

      for (LocalOrdinal r = 0; r < nRows; ++r) {
        const GlobalOrdinal grow = opMap_->getGlobalElement(r);
        Scalar val;

        if (grow == 0) {
          // Left boundary: stencil (2, -1)
          val = 2.0 * col[r + 1 - off] - col[r + 2 - off];
        } else if (grow == opMap_->getMaxAllGlobalIndex()) {
          // Right boundary: stencil (-1, 2)
          val = -col[r - off] + 2.0 * col[r + 1 - off];
        } else {
          // Interior: stencil (-1, 2, -1)
          val = -col[r - off] + 2.0 * col[r + 1 - off] - col[r + 2 - off];
        }

        if (beta == STS::zero())
          Y.replaceLocalValue(r, c, alpha * val);
        else
          Y.sumIntoLocalValue(r, c, alpha * val);
      }
    }
  }

 private:
  Teuchos::RCP<const map_type>    opMap_;      // non-overlapping (owned) map
  Teuchos::RCP<const map_type>    redistMap_;  // overlapping (ghosted) map
  Teuchos::RCP<const import_type> importer_;   // owned -> ghosted
};
