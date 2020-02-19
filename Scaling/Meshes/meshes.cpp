#include <cmath>
#include <dolfinx.h>
#include <dolfinx/common/MPI.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace dolfinx;

int main(int argc, char *argv[]) {
  common::SubSystemsManager::init_logging(argc, argv);
  common::SubSystemsManager::init_petsc(argc, argv);

  std::vector<int> num_core_list = {20};
  const int local_size = 100000;
  const unsigned int num_refines = 3;

  for (auto num_cores : num_core_list) {

    // Number of cells in each direction
    long unsigned int Ni = round(pow(
        (num_cores * local_size) / (6 * pow(8, num_refines - 1)), 1.0 / 3.0));

    // Create mesh using all processes
    std::array<Eigen::Vector3d, 2> pt{Eigen::Vector3d(0.0, 0.0, 0.0),
                                      Eigen::Vector3d(1.0, 1.0, 1.0)};

    // generate mesh
    auto mesh = std::make_shared<mesh::Mesh>(generation::BoxMesh::create(
        MPI_COMM_WORLD, pt, {{Ni, Ni, Ni}}, mesh::CellType::tetrahedron,
        mesh::GhostMode::none));

    // refine mesh num_refines times
    for (unsigned int i = 0; i != num_refines; ++i) {
      mesh = std::make_shared<mesh::Mesh>(refinement::refine(*mesh, false));
    }

    // XDMF file name
    std::stringstream filename;
    filename << "../files/mesh" << std::to_string(num_cores) << ".xdmf";

    // save mesh in XMDF format
    io::XDMFFile file(MPI_COMM_WORLD, filename.str());
    file.write(*mesh);
  }

  return 0;
}
