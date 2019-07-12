from dolfin import (MPI, UnitCubeMesh)
from dolfin.io import XDMFFile
from dolfin.cpp.refinement import refine

comm = MPI.comm_world

Plist = [32, 64, 128, 192, 256, 320]
Nc = 100000
R = 3

for P in Plist:
        Ni = round((P*Nc/(6*(8**(R-1))))**(1/3))

        mesh = UnitCubeMesh(MPI.comm_world, Ni, Ni, Ni)
        for i in range(R-1):
                mesh = refine(mesh, False)

        filename = "mesh" + str(P) + "_100.xdmf"
        with XDMFFile(mesh.mpi_comm(), filename, XDMFFile.Encoding.HDF5) as xdmf:
                xdmf.write(mesh)

        # Number of Processors
        print(mesh.num_cells())
        print(Nc*P/4)
