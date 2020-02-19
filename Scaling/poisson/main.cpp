#include "poisson.h"
#include <cfloat>
#include <dolfinx.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/timing.h>
#include <dolfinx/function/Constant.h>

using namespace dolfinx;

int main(int argc, char *argv[]) {
  common::SubSystemsManager::init_logging(argc, argv);
  common::SubSystemsManager::init_petsc(argc, argv);

  mesh::CellType::Type cell_type;
  EigenRowArrayXXd points;
  EigenRowArrayXXi64 cells;
  std::vector<std::int64_t> global_cell_indices;

  std::stringstream ss;
  ss << "../../Meshes/files/mesh" << std::to_string(mpi_size) << ".xdmf";

  // Read in mesh in mesh data from XDMF file
  io::XDMFFile infile(MPI_COMM_WORLD, filename);
  const auto [cell_type, points, cells, global_cell_indices] =
      infile.read_mesh_data(MPI_COMM_WORLD);

  // Partition mesh into nparts using local mesh data and subset of
  // communicators
  int nparts = mpi_size;
  mesh::PartitionData cell_partition = mesh::Partitioning::partition_cells(
      subset_comm, nparts, cell_type, cells, mesh::Partitioner::scotch);

  // Build mesh from local mesh data, ghost mode, and provided cell partition
  mesh::GhostMode ghost_mode = mesh::GhostMode::none;
  auto new_mesh =
      std::make_shared<mesh::Mesh>(mesh::Partitioning::build_from_partition(
          mpi_comm.comm(), cell_type, points, cells, global_cell_indices,
          ghost_mode, cell_partition));

  auto V = fem::create_functionspace(poisson_functionspace_create, mesh);

  std::shared_ptr<fem::Form> a =
      fem::create_form(poisson_bilinearform_create, {V, V});

  std::shared_ptr<fem::Form> L =
      fem::create_form(poisson_linearform_create, {V});

  auto f = std::make_shared<function::Function>(V);
  auto g = std::make_shared<function::Function>(V);

  // Attach 'coordinate mapping' to mesh
  auto cmap = a->coordinate_mapping();
  mesh->geometry().coord_mapping = cmap;

  auto u0 = std::make_shared<function::Function>(V);

  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1> bdofs =
      fem::locate_dofs_geometrical({*V}, [](auto &x) {
        return (x.row(0) < DBL_EPSILON or x.row(0) > 1.0 - DBL_EPSILON);
      });

  std::vector<std::shared_ptr<const fem::DirichletBC>> bc = {
      std::make_shared<fem::DirichletBC>(u0, bdofs)};

  f->interpolate([](auto &x) {
    auto dx = Eigen::square(x - 0.5);
    return 10.0 * Eigen::exp(-(dx.row(0) + dx.row(1)) / 0.02);
  });

  g->interpolate([](auto &x) { return Eigen::sin(5 * x.row(0)); });
  L->set_coefficients({{"f", f}, {"g", g}});

  // Prepare and set Constants for the bilinear form
  auto kappa = std::make_shared<function::Constant>(2.0);
  a->set_constants({{"kappa", kappa}});

  function::Function u(V);
  la::PETScMatrix A = fem::create_matrix(*a);
  la::PETScVector b(*L->function_space(0)->dofmap()->index_map);

  MatZeroEntries(A.mat());
  dolfinx::fem::assemble_matrix(A.mat(), *a, bc);
  dolfinx::fem::add_diagonal(A.mat(), *V, bc);
  MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

  VecSet(b.vec(), 0.0);
  VecGhostUpdateBegin(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
  dolfinx::fem::assemble_vector(b.vec(), *L);
  dolfinx::fem::apply_lifting(b.vec(), {a}, {{bc}}, {}, 1.0);
  VecGhostUpdateBegin(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  VecGhostUpdateEnd(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  dolfinx::fem::set_bc(b.vec(), bc, nullptr);

  la::PETScKrylovSolver lu(MPI_COMM_WORLD);
  la::PETScOptions::set("ksp_type", "preonly");
  la::PETScOptions::set("pc_type", "lu");
  lu.set_from_options();

  lu.set_operator(A.mat());
  lu.solve(u.vector().vec(), b.vec());

  io::VTKFile file("u.pvd");
  file.write(u);

  // Display timings
  dolfinx::list_timings(MPI_COMM_WORLD, {dolfinx::TimingType::wall});

  return 0;
}
