# Matrix-Free Belos Solve (No Tpetra)

Minimal example of a **matrix-free** iterative solve using
[Trilinos/Belos](https://trilinos.github.io/belos.html) with fully custom data
structures -- no Tpetra vectors, maps, or operators.

## Idea

Belos (CG, GMRES, ...) is templated on the scalar, multivector, and operator
types. By specializing `Belos::MultiVecTraits` and `Belos::OperatorTraits` for
our own classes we can plug any application data directly into Belos without
converting to/from Tpetra objects.

| Belos concept | Our implementation | Purpose |
|---|---|---|
| MultiVector | `SimpleMultiVec` | Column-major storage via `shared_ptr<vector<double>>` per column; supports views (shared columns, no copy). Maps to `field_impl` in specfem++. |
| Operator | `SimpleOperator` (abstract) | Pure-virtual `apply(X, Y)`. Maps to `compute_stiffness` in specfem++. |
| -- | `Laplacian1D` | Concrete test operator: tridiagonal (-1, 2, -1) stencil. |
| -- | `SpectralElementOperator2D` | Skeleton gather-local_action-scatter operator showing the SE pattern. |

## Files

```
include/
  SimpleMultiVec.hpp        -- multivector (field storage)
  SimpleOperator.hpp        -- operator base + Laplacian1D + SE2D skeleton
  BelosSimpleAdapters.hpp   -- MultiVecTraits / OperatorTraits specializations
src/
  main.cpp                  -- solve A x = b with Belos PseudoBlockCG
CMakeLists.txt
```

## Build

```bash
mkdir build && cd build
cmake .. -DTrilinos_DIR=/path/to/TrilinosConfig.cmake
make
```

## Run

```bash
mpirun -np 1 ./matrix_free_solve
```

Solves a 100-point 1-D Laplacian with CG to tolerance 1e-10 and verifies the
solution against the known answer.
