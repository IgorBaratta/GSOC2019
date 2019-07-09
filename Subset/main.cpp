#include <cfloat>
#include <cmath>
#include <dolfin.h>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/PartitionData.h>
#include <dolfin/mesh/Partitioning.h>
#include <iostream>

using namespace dolfin;

int main() {
  int argc = 0;
  char **argv = nullptr;
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  auto mpi_comm = dolfin::MPI::Comm(MPI_COMM_WORLD);
  int mpi_size = mpi_comm.size();

  // Create sub-communicator
  int subset_size = (mpi_size > 1) ? ceil(mpi_size / 2) : 1;
  MPI_Comm subset_comm = dolfin::MPI::SubsetComm(MPI_COMM_WORLD, subset_size);

  // Create mesh using all processes
  std::array<Eigen::Vector3d, 2> pt{Eigen::Vector3d(0.0, 0.0, 0.0),
                                    Eigen::Vector3d(1.0, 1.0, 0.0)};
  auto mesh = std::make_shared<mesh::Mesh>(generation::RectangleMesh::create(
      MPI_COMM_WORLD, pt, {{64, 64}}, mesh::CellType::Type::triangle,
      mesh::GhostMode::none));

  // Save mesh in XDMF format
  io::XDMFFile file(MPI_COMM_WORLD, "mesh.xdmf");
  file.write(*mesh);

  mesh::CellType::Type cell_type;
  EigenRowArrayXXd points;
  EigenRowArrayXXi64 cells;
  std::vector<std::int64_t> global_cell_indices;

  // Save mesh in XDMF format
  io::XDMFFile infile(MPI_COMM_WORLD, "mesh.xdmf");
  std::tie(cell_type, points, cells, global_cell_indices) =
      infile.read_mesh_data(subset_comm);

  // Partition mesh into nparts using local mesh data and subset of
  // communicators
  int nparts = mpi_size;
  std::string partitioner = "SCOTCH";
  mesh::PartitionData cell_partition = mesh::Partitioning::partition_cells(
      subset_comm, nparts, cell_type, cells, partitioner);

  // Build mesh from local mesh data, ghost mode, and provided cell partition
  auto ghost_mode = mesh::GhostMode::none;
  auto new_mesh =
      std::make_shared<mesh::Mesh>(mesh::Partitioning::build_from_partition(
          mpi_comm.comm(), cell_type, cells, points, global_cell_indices,
          ghost_mode, cell_partition));

  assert(mesh->num_entities_global(0) == new_mesh->num_entities_global(0));

  std::cout << "/* message */" << new_mesh->num_entities_global(0) << '\n';
  std::cout << "/* message */" << mesh->num_entities_global(0) << '\n';

  assert(mesh->num_entities_global(1) == new_mesh->num_entities_global(1));

  return 0;
}
