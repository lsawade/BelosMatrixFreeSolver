// ---------------------------------------------------------------------
// Matrix-free Belos solve – NO Tpetra.
//
// Tests ALL available Belos solver managers against a 1D Laplacian
// and prints a summary comparison table.
// ---------------------------------------------------------------------

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <mpi.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosSolverManager.hpp>

// All available solver managers
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosPCPGSolMgr.hpp>
#include <BelosRCGSolMgr.hpp>
#include <BelosPseudoBlockStochasticCGSolMgr.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
// GCRODRSolMgr omitted: requires Range1D overloads in MultiVecTraits
// GmresPolySolMgr omitted: requires Range1D overloads and SolverFactory
#include <BelosBiCGStabSolMgr.hpp>
#include <BelosTFQMRSolMgr.hpp>
#include <BelosPseudoBlockTFQMRSolMgr.hpp>
#include <BelosMinresSolMgr.hpp>
#include <BelosFixedPointSolMgr.hpp>
#include <BelosLSQRSolMgr.hpp>

#include "SimpleMultiVec.hpp"
#include "SimpleOperator.hpp"
#include "BelosSimpleAdapters.hpp"

using ST = double;
using MV = SimpleMultiVec;
using OP = SimpleOperator;
using problem_t = Belos::LinearProblem<ST, MV, OP>;
using solver_t  = Belos::SolverManager<ST, MV, OP>;

struct SolveResult {
  std::string name;
  bool        converged;
  int         iterations;
  double      error;
  std::string failReason;
};

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

  // ---- Build operator and RHS (once, shared by all solvers) ----
  RCP<OP> A = rcp(new Laplacian1D(n));

  // b chosen so x_exact = [1, 1, ..., 1]^T
  RCP<MV> b = rcp(new MV(n, 1));
  (*b)(0, 0)     = 1.0;
  (*b)(n - 1, 0) = 1.0;

  // ---- Common solver parameters (quiet output) ----
  RCP<ParameterList> params = rcp(new ParameterList());
  params->set("Maximum Iterations",   maxIters);
  params->set("Convergence Tolerance", tol);
  params->set("Verbosity",            Belos::Errors);

  // ---- Results collector ----
  std::vector<SolveResult> results;

  // Helper: solve, compute error, record result
  auto runSolve = [&](const std::string& name,
                      RCP<solver_t> solver,
                      RCP<MV> x) {
    try {
      Belos::ReturnType ret = solver->solve();
      double errNorm = 0.0;
      for (int i = 0; i < n; ++i) {
        double e = (*x)(i, 0) - 1.0;
        errNorm += e * e;
      }
      errNorm = std::sqrt(errNorm);
      results.push_back({name, ret == Belos::Converged,
                         solver->getNumIters(), errNorm, ""});
    } catch (const std::exception& e) {
      results.push_back({name, false, 0, -1.0, e.what()});
    }
  };

  // Macro: create fresh x, problem, solver; run and record
  #define TRY_SOLVER(SolverType, Name, Params)                        \
    {                                                                  \
      cout << "  Running " << Name << "..." << endl;                   \
      auto x    = rcp(new MV(n, 1));                                   \
      auto prob = rcp(new problem_t(A, x, b));                         \
      prob->setProblem();                                               \
      try {                                                             \
        auto s = rcp(new SolverType<ST, MV, OP>(prob, Params));        \
        runSolve(Name, s, x);                                          \
      } catch (const std::exception& e) {                              \
        results.push_back({Name, false, 0, -1.0, e.what()});          \
      }                                                                 \
    }

  cout << "=== Testing all Belos solvers (n=" << n
       << ", tol=" << tol << ") ===" << endl;

  // --- CG family (SPD) ---
  TRY_SOLVER(Belos::PseudoBlockCGSolMgr,           "PseudoBlockCG",   params)
  TRY_SOLVER(Belos::BlockCGSolMgr,                 "BlockCG",         params)
  {
    auto p = rcp(new ParameterList(*params));
    p->set("Num Recycled Blocks", 5);
    TRY_SOLVER(Belos::PCPGSolMgr,                  "PCPG",            p)
  }
  {
    auto p = rcp(new ParameterList(*params));
    p->set("Num Recycled Blocks", 5);
    TRY_SOLVER(Belos::RCGSolMgr,                   "RCG",             p)
  }
  TRY_SOLVER(Belos::PseudoBlockStochasticCGSolMgr, "StochasticCG",    params)

  // --- GMRES family ---
  TRY_SOLVER(Belos::BlockGmresSolMgr,              "BlockGMRES",      params)
  TRY_SOLVER(Belos::PseudoBlockGmresSolMgr,        "PseudoBlockGMRES",params)
  // GCRODRSolMgr omitted: requires Range1D overloads
  // GmresPolySolMgr omitted: requires Range1D overloads and SolverFactory

  // --- Other Krylov methods ---
  TRY_SOLVER(Belos::BiCGStabSolMgr,                "BiCGStab",        params)
  TRY_SOLVER(Belos::TFQMRSolMgr,                   "TFQMR",           params)
  TRY_SOLVER(Belos::PseudoBlockTFQMRSolMgr,        "PseudoBlockTFQMR",params)
  TRY_SOLVER(Belos::MinresSolMgr,                  "MINRES",          params)
  TRY_SOLVER(Belos::FixedPointSolMgr,              "FixedPoint",      params)
  TRY_SOLVER(Belos::LSQRSolMgr,                    "LSQR",            params)

  #undef TRY_SOLVER

  // ---- Summary table ----
  cout << "\n" << std::string(72, '=') << endl;
  cout << std::left  << std::setw(22) << "Solver"
       << std::right << std::setw(12) << "Converged"
       << std::setw(10) << "Iters"
       << std::setw(18) << "||x - x*||_2"
       << endl;
  cout << std::string(72, '-') << endl;

  for (const auto& r : results) {
    cout << std::left << std::setw(22) << r.name;
    if (r.failReason.empty()) {
      cout << std::right
           << std::setw(12) << (r.converged ? "yes" : "NO")
           << std::setw(10) << r.iterations
           << std::setw(18) << std::scientific << std::setprecision(2)
           << r.error;
    } else {
      std::string msg = r.failReason.substr(0, 45);
      cout << "  FAILED: " << msg;
    }
    cout << endl;
  }
  cout << std::string(72, '=') << endl;

  MPI_Finalize();
  return EXIT_SUCCESS;
}
