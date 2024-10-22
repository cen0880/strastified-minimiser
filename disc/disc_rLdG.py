from dolfin import *
from mshr import *
import numpy as np
mesh = generate_mesh(Circle(Point(0,0),1),256)
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
#Create intial conditions
#read theta and construct initial condition with q11 = s*cos(2*theta), q12 = s*sin(2*theta)
V = FunctionSpace(mesh, 'Lagrange', 1)
theta = Function(V)
input_file = HDF5File(mesh.mpi_comm(), "./image/angle_TRI.h5", "r")
input_file.read(theta, "solution")
input_file.close()
qa = q.vector().get_local()
qa[::2] = s0*np.cos(2*theta.vector().get_local())
qa[1::2] = s0*np.sin(2*theta.vector().get_local())
q.vector().set_local(qa)

#read input file
#input_file = HDF5File(mesh.mpi_comm(), "./image/q_lambda_square_500_RING.h5", "r")
#input_file = HDF5File(mesh.mpi_comm(), "./image/triangle_disc.h5", "r")
#input_file.read(q, "solution")
#input_file.close()

#V = FunctionSpace(mesh, P1)
#u = Function(V)
#input_file = HDF5File(mesh.mpi_comm(), "./q1_WORS.h5", "r")
#input_file.read(u, "solution")
#input_file.close()
#ua = u.vector().get_local()
#qa = q.vector().get_local()
#qa[::2] = ua
#qa[1::2] = np.zeros(ua.size)

F0 = lambda_square*(q11*q11 + q12*q12 - B*B/4/C/C)*q11*v11*dx + inner(grad(q11), grad(v11))*dx
F1 = lambda_square*(q11*q11 + q12*q12 - B*B/4/C/C)*q12*v12*dx + inner(grad(q12), grad(v12))*dx
F = F0 + F1

# Define Dirichlet boundary (x = -1 or x = 1)
homeo = interpolate(Expression(("s0*cos(2.0*atan2(x[1],x[0]))","s0*sin(2.0*atan2(x[1],x[0]))"), degree = 1, s0 = s0),ME)   
bc = DirichletBC(ME, homeo, 'on_boundary')    

a = derivative(F, q)
problem = NonlinearVariationalProblem(F, q, bc, a)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['linear_solver'] = "lu"
prm['newton_solver']['absolute_tolerance'] = 1E-13
prm['newton_solver']['relative_tolerance'] = 1E-15
prm['newton_solver']['maximum_iterations'] = 50
prm['newton_solver']['relaxation_parameter'] = 1.0
solver.solve()

# calculate energy
E = assemble(1.0/2.0*lambda_square*pow(q11*q11 + q12*q12 - B*B/4/C/C,2)*dx + 1.0/2.0*(inner(grad(q11), grad(q11))+inner(grad(q12),grad(q12)))*dx)
print("Energy = " + str(E))
# largest eigenvalue
# a = inner(grad(q), grad(v))*dx

# calculate smallest k eigenvalues
#k = 4
#A = PETScMatrix()
#assemble(a,tensor = A)
#bc.apply(A)
#eigensolver = SLEPcEigenSolver(A)

# # assemble(a, tensor = A)
# # eigensolver = SLEPcEigenSolver(A,bc)
# # eigensolver.parameters['spectrum']= 'largest real'

#eigensolver.parameters['spectrum']= 'smallest real'
#eigensolver.solve(k)
#for n in range(0,k):
#    r, c, rx, cx = eigensolver.get_eigenpair(n)
#    print('eigenvalue' + str(n) + ' ' + str(r))

# output .h5 data file
output_file = HDF5File(mesh.mpi_comm(), "./image/q_lambda_square_"+str(lambda_square)+"_TRI.h5", "w")
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
vtkfile = File('./image/s'+str(lambda_square)+"_TRI.pvd")
vtkfile << sf

# output a .vtu file for nematic direction visualization in Paraview
angle = np.arctan2(u2a,u1a)/2.0
D = VectorFunctionSpace(mesh,'Lagrange',1,dim=2)
dd = Function(D)
da = dd.vector().get_local()
da[::2] = np.cos(angle)*s
da[1::2] = np.sin(angle)*s
dd.vector().set_local(da)
vtkfile = File('./image/direction'+str(lambda_square)+"_TRI.pvd")
vtkfile << dd


