from dolfin import *
from mshr import *
import numpy as np
domain_vertices = [Point(1.0,0.0),
        Point(0.0,1.0),
        Point(-1.0,0.0),
        Point(0.0,-1.0),
        Point(1.0,0.0)]
# Generate mesh and plot
domain = Polygon(domain_vertices)
mesh = generate_mesh(domain, 64)
# Model parameters
B = 0.64*1e4
C = 0.35*1e4
s0 = B/C/2.0
lambda_square = 1000
# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
# Create mesh and build function space
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = FunctionSpace(mesh, P1*P1)
# Define trial and test functions
v11, v12 = TestFunctions(ME)
# Define functions
q = Function(ME)  # current solution
# Split mixed functions
q11, q12 = split(q)

#read input file
input_file.read(q, "solution")
input_file.close()
F0 = lambda_square*(q11*q11 + q12*q12 - B*B/4/C/C)*q11*v11*dx + inner(grad(q11), grad(v11))*dx
F1 = lambda_square*(q11*q11 + q12*q12 - B*B/4/C/C)*q12*v12*dx + inner(grad(q12), grad(v12))*dx
F = F0 + F1

def C1(x, on_boundary):
    return x[0] > 0.0 and x[1] > 0.0 and on_boundary
bc1 = DirichletBC(ME, (0.0,-s0), C1)
def C2(x, on_boundary):
    return x[0] < 0.0 and x[1] > 0.0 and on_boundary
bc2 = DirichletBC(ME, (0.0,s0), C2)
def C3(x, on_boundary):
    return x[0] < 0.0 and x[1] < 0.0 and on_boundary
bc3 = DirichletBC(ME, (0.0,-s0), C3)
def C4(x, on_boundary):
    return x[0] > 0.0 and x[1] < 0.0 and on_boundary
bc4 = DirichletBC(ME, (0.0,s0), C4)  

bc = [bc1,bc2,bc3,bc4]

a = derivative(F, q)
problem = NonlinearVariationalProblem(F, q, bc, a)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['linear_solver'] = "lu"
prm['newton_solver']['absolute_tolerance'] = 1E-2
prm['newton_solver']['relative_tolerance'] = 1E-2
prm['newton_solver']['maximum_iterations'] = 150
prm['newton_solver']['relaxation_parameter'] = 1.0
solver.solve()

# calculate energy
E = assemble(1.0/2.0*lambda_square*pow(q11*q11 + q12*q12 - B*B/4/C/C,2)*dx + 1.0/2.0*(inner(grad(q11), grad(q11))+inner(grad(q12),grad(q12)))*dx)
print("Energy = " + str(E))

# output .h5 data file
output_file = HDF5File(mesh.mpi_comm(), "./image/q_lambda_square_"+str(lambda_square)+"_J-I.h5", "w")
output_file.write(q, "solution")
output_file.close()

# output a .vtu file for nematic order visualization in Paraview
qa = q.vector().get_local()
u1a = qa[::2]
u2a = qa[1::2]
s = np.sqrt(u1a*u1a+u2a*u2a)
V = FunctionSpace(mesh, 'Lagrange', 1)
sf = Function(V)
sf.vector().set_local(s)
vtkfile = File('./image/s'+str(lambda_square)+"_J-I.pvd")
vtkfile << sf

# output a .vtu file for nematic direction visualization in Paraview
angle = np.arctan2(u2a,u1a)/2.0
D = VectorFunctionSpace(mesh,'Lagrange',1,dim=2)
dd = Function(D)
da = dd.vector().get_local()
da[::2] = np.cos(angle)*s
da[1::2] = np.sin(angle)*s
dd.vector().set_local(da)
vtkfile = File('./image/direction'+str(lambda_square)+"_J-I.pvd")
vtkfile << dd