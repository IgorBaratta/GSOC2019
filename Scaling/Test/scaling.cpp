#include <cfloat>
#include <dolfin.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/timing.h>
#include <dolfin/mesh/Partitioning.h>
#include <iostream>
#include <sstream>
#include <string>

using namespace dolfin;

int main(int argc, char *argv[]) {
  common::SubSystemsManager::init_logging(argc, argv);
  common::SubSystemsManager::init_petsc(argc, argv);

  auto mpi_comm = dolfin::MPI::Comm(MPI_COMM_WORLD);
  int mpi_size = dolfin::MPI::size(mpi_comm.comm());

  // ----------------------------------------------- //
  // Create sub-communicator
  dolfin::common::Timer tx("___ Create sub-communicator");
  // int subset_size = (mpi_size > 1) ? ceil(mpi_size / 2) : 1;
  int subset_size = mpi_size;
  MPI_Comm subset_comm = dolfin::MPI::SubsetComm(MPI_COMM_WORLD, subset_size);
  tx.stop();

  // ----------------------------------------------- //
  // mesh data
  mesh::CellType::Type cell_type;
  EigenRowArrayXXd points;
  EigenRowArrayXXi64 cells;
  std::vector<std::int64_t> global_cell_indices;

  std::stringstream ss;
  ss << "../../Meshes/files/mesh" << std::to_string(mpi_size) << ".xdmf";

  std::string filename = ss.str();

  // ----------------------------------------------- //
  // Read mesh data from XDMF file
  dolfin::common::Timer t0("___ Read Mesh data");
  io::XDMFFile infile(MPI_COMM_WORLD, filename);
  std::tie(cell_type, points, cells, global_cell_indices) =
      infile.read_mesh_data(subset_comm);
  t0.stop();

  // ----------------------------------------------- //
  // Partition mesh into nparts using local mesh data and a
  // communicators
  dolfin::common::Timer t1("___ Partition Mesh");
  int nparts = mpi_size;
  std::string partitioner = "SCOTCH";
  mesh::PartitionData cell_partition = mesh::Partitioning::partition_cells(
      subset_comm, nparts, cell_type, cells, partitioner);
  t1.stop();

  // ----------------------------------------------- //
  // Build mesh from local mesh data, ghost mode, and provided cell partition
  dolfin::common::Timer t2("___ Build Distributed Mesh");
  auto ghost_mode = mesh::GhostMode::none;
  auto new_mesh =
      std::make_shared<mesh::Mesh>(mesh::Partitioning::build_from_partition(
          mpi_comm.comm(), cell_type, cells, points, global_cell_indices,
          ghost_mode, cell_partition));

  t2.stop();

  if (mpi_comm.rank() == 0) {
    std::cout << "___ Partitioner: " << partitioner << std::endl;
    std::cout << "___ Comm Size: " << mpi_size << std::endl;
    std::cout << "___ Subset size: " << subset_size << std::endl;
  }

  // Display timings
  dolfinx::list_timings(MPI_COMM_WORLD, {dolfinx::TimingType::wall});

  return 0;
}
