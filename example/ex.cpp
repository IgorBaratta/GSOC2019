#include <iostream>

// #include "kaHIP_interface.h"
#include <mpi.h>
#include <parhip_interface.h>

int main(int argc, char **argv) {

  // Initialize the MPI environment
  MPI_Init(&argc, &argv);
  idxtype np = 4;
  idxtype xadj_[3];
  idxtype adjncy_[5];
  int rank, n;
  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
    std::cout << "Comm Size: " << n;

  if (rank == 0) {
    xadj_[0] = 0;
    xadj_[1] = 2;
    xadj_[2] = 5;
    adjncy_[0] = 4;
    adjncy_[1] = 1;
    adjncy_[2] = 0;
    adjncy_[3] = 5;
    adjncy_[4] = 2;
  }
  if (rank == 1) {
    xadj_[0] = 0;
    xadj_[1] = 3;
    xadj_[2] = 5;
    adjncy_[0] = 1;
    adjncy_[1] = 6;
    adjncy_[2] = 3;
    adjncy_[3] = 2;
    adjncy_[4] = 7;
  }
  if (rank == 2) {
    xadj_[0] = 0;
    xadj_[1] = 2;
    xadj_[2] = 5;
    adjncy_[0] = 5;
    adjncy_[1] = 0;
    adjncy_[2] = 6;
    adjncy_[3] = 1;
    adjncy_[4] = 4;
  }
  if (rank == 3) {
    xadj_[0] = 0;
    xadj_[1] = 3;
    xadj_[2] = 5;
    adjncy_[0] = 7;
    adjncy_[1] = 2;
    adjncy_[2] = 5;
    adjncy_[3] = 3;
    adjncy_[4] = 6;
  }
  idxtype *xadj = xadj_;
  idxtype *adjncy = adjncy_;

  idxtype vtxdist_[] = {0, 2, 4, 6, 8};
  idxtype *vtxdist = vtxdist_;

  int nparts = n;

  idxtype *vwgt{nullptr};
  idxtype *adjcwgt{nullptr};

  // The amount of imbalance that is allowed. (3%)
  double imbalance = 0.03;

  // Suppress output from the partitioning library.
  bool suppress_output = false;

  int mode = ULTRAFASTMESH;
  int seed = 0;
  int edgecut = 0;

  idxtype *part;

  ParHIPPartitionKWay(vtxdist, xadj, adjncy, vwgt, adjcwgt, &nparts, &imbalance,
                      suppress_output, seed, mode, &edgecut, part, &comm);

  MPI_Finalize();
  return 0;
}
