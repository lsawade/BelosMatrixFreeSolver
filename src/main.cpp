// ---------------------------------------------------------------------
// Matrix-free Belos solve – NO Tpetra.
//
// Belos drives the Krylov iteration (GMRES) using only:
//   - SimpleMultiVec       (your own vector type)
//   - SimpleOperator       (your own matrix-free operator)
//   - BelosSimpleAdapters  (MultiVecTraits + OperatorTraits)
//
// The operator and vector never touch Tpetra::Map, Tpetra::MultiVector,
// or Tpetra::Operator.  Data lives in plain arrays that match the
// layout of a spectral element code: u(nglob), mapping(nspec,ngll,ngll).
//
// Run:   ./matrix_free_solve
// ---------------------------------------------------------------------

#include <iostream>
#include <cmath>
#include <mpi.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>

#include "SimpleMultiVec.hpp"
#include "SimpleOperator.hpp"
#include "BelosSimpleAdapters.hpp"

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using std::cout;
  using std::endl;

  // ---- Problem parameters ----
  const int    n        = 100;
  const double tol      = 1.0e-10;
  const int    maxIters = 400;

  cout << "=== Matrix-free Belos solve (no Tpetra) ===" << endl;
  cout << "  n        = " << n        << endl;
  cout << "  tol      = " << tol      << endl;
  cout << "  maxIters = " << maxIters << endl;

  // ---- Build the operator (1D Laplacian for proof of concept) ----
  RCP<SimpleOperator> A = rcp(new Laplacian1D(n));

  // ---- RHS: b chosen so x_exact = [1, 1, ..., 1]^T ----
  //  For the 1D Laplacian with Dirichlet BCs:
  //    b[0]   = 1   (boundary: 2·1 - 1)
  //    b[i]   = 0   (interior: -1 + 2 - 1)
  //    b[n-1] = 1   (boundary: -1 + 2·1)
  RCP<SimpleMultiVec> b = rcp(new SimpleMultiVec(n, 1));
  (*b)(0, 0)     = 1.0;
  (*b)(n - 1, 0) = 1.0;

  // ---- Initial guess x = 0 ----
  RCP<SimpleMultiVec> x = rcp(new SimpleMultiVec(n, 1));

  // ---- Set up the Belos linear problem ----
  using problem_type = Belos::LinearProblem<double, SimpleMultiVec, SimpleOperator>;
  RCP<problem_type> problem = rcp(new problem_type(A, x, b));

  bool set = problem->setProblem();
  if (!set) {
    cout << "ERROR: Belos::LinearProblem::setProblem() failed." << endl;
    return EXIT_FAILURE;
  }

  // ---- Configure the Belos solver (GMRES) ----
  // For an SPD system like the Laplacian, you could also use "Block CG"
  // or "Pseudo Block CG".
  RCP<ParameterList> params = rcp(new ParameterList());
  params->set("Maximum Iterations",    maxIters);
  params->set("Convergence Tolerance", tol);
  params->set("Verbosity",
              Belos::Errors + Belos::Warnings +
              Belos::TimingDetails + Belos::StatusTestDetails);
  params->set("Output Frequency", 1);

  RCP<Belos::SolverManager<double, SimpleMultiVec, SimpleOperator>> solver =
      rcp(new Belos::PseudoBlockCGSolMgr<double, SimpleMultiVec, SimpleOperator>(
          problem, params));

  // ---- Solve ----
  cout << "\nSolving..." << endl;

  Belos::ReturnType result = solver->solve();

  cout << "\nBelos "
       << (result == Belos::Converged ? "converged" : "DID NOT converge")
       << " in " << solver->getNumIters() << " iterations." << endl;

  // ---- Check against x_exact = 1 ----
  double errNorm = 0.0;
  for (int i = 0; i < n; ++i) {
    double e = (*x)(i, 0) - 1.0;
    errNorm += e * e;
  }
  errNorm = std::sqrt(errNorm);

  cout << "  ||x - x_exact||_2 = " << errNorm << endl;

  if (result == Belos::Converged && errNorm < 1.0e-6)
    cout << "\nTEST PASSED" << endl;
  else
    cout << "\nTEST FAILED" << endl;

  MPI_Finalize();
  return (result == Belos::Converged) ? EXIT_SUCCESS : EXIT_FAILURE;
}
