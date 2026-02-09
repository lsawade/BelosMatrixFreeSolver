cmake_minimum_required(VERSION 3.20)
project(MatrixFreeSolve CXX)

# ---------- Find pre-installed Trilinos ----------
# Pass -DTrilinos_DIR=<path-to-TrilinosConfig.cmake> if it is not in a
# standard location (e.g. -DTrilinos_DIR=/opt/trilinos/lib/cmake/Trilinos).
find_package(MPI REQUIRED)
find_package(Trilinos REQUIRED)
message(STATUS "Found Trilinos ${Trilinos_VERSION}")
message(STATUS "Trilinos libraries: ${Trilinos_LIBRARIES}")

# ---------- Build the example ----------
add_executable(matrix_free_solve
  src/main.cpp
)

target_include_directories(matrix_free_solve PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${Trilinos_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
)

target_link_libraries(matrix_free_solve PRIVATE
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  MPI::MPI_CXX
)
