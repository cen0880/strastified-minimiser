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
# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
# Create mesh and build function space
V = FunctionSpace(mesh, 'Lagrange', 1)
# Define the test and trial functions
phi = Function(V)
v = TestFunction(V)

N = 3 # number of interior defects
ax = [0] * N 
ay = [0] * N
d = [0] * N
theta = [0] * N
K = 4 # number of domain corners
c = [0] * K
g = [0] * K
name = "J-I" # name of state
d = [-1.0, 1.0, -1.0] # winding number of interior defects
ax = [-0.35, 0.5, 0.2] # location x of interior defects
ay = [0.35, -0.2, -0.5] # location y of interior defects
c = [-1.0,1.0,1.0,1.0] # -c/2 winding number of corner defects

# calculate g
g[0] = -np.pi/2.0
for i in range(K-1):
    g[i+1] = g[i] - 2.0*c[i+1]*np.pi/2.0

# calculate phi0
for i in range(N):
    theta[i] = np.pi-np.arctan2(-ay[i],1.0-ax[i])
class MyExpression0(UserExpression):
    def eval(self, value, x):
        value[0] = 0
        for i in range(N):
            value[0] = value[0] - d[i]*(np.arctan2(sin(theta[i])*(x[0]-ax[i]) + cos(theta[i])*(x[1]-ay[i]),cos(theta[i])*(x[0]-ax[i])-sin(theta[i])*(x[1]-ay[i])) - theta[i])
phi0 = interpolate(MyExpression0(),V)

# boundary condition
def C1(x, on_boundary):
    return x[0] > 0.0 and x[1] > 0.0 and on_boundary
bc1 = DirichletBC(V, phi0 + g[0], C1)
def C2(x, on_boundary):
    return x[0] < 0.0 and x[1] > 0.0 and on_boundary
bc2 = DirichletBC(V, phi0 + g[1], C2)
def C3(x, on_boundary):
    return x[0] < 0.0 and x[1] < 0.0 and on_boundary
bc3 = DirichletBC(V, phi0 + g[2], C3)
def C4(x, on_boundary):
    return x[0] > 0.0 and x[1] < 0.0 and on_boundary
bc4 = DirichletBC(V, phi0 + g[3], C4)  
bc = [bc1,bc2,bc3,bc4]

# solve phi
F = inner(grad(phi), grad(v))*dx
a = derivative(F, phi)
problem = NonlinearVariationalProblem(F, phi, bc, a)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['linear_solver'] = "lu"
prm['newton_solver']['absolute_tolerance'] = 1E-4
prm['newton_solver']['relative_tolerance'] = 1E-7
prm['newton_solver']['maximum_iterations'] = 10
prm['newton_solver']['relaxation_parameter'] = 1.0
solver.solve()

# output a .vtu file for nematic direction visualization in Paraview
anglea = (phi.vector().get_local()-phi0.vector().get_local())/2.0
angle = Function(V)
angle.vector().set_local(anglea)
D = VectorFunctionSpace(mesh,'Lagrange',1,dim=2)
dd = Function(D)
da = dd.vector().get_local()
da[::2] = np.cos(anglea)
da[1::2] = np.sin(anglea)
dd.vector().set_local(da)
vtkfile = File('./image/direction_'+name+'.pvd')
vtkfile << dd

# output .h5 data file for the initial condition of rLdG
P1 = FiniteElement("Lagrange", mesh.ufl_cell(),1)
ME = FunctionSpace(mesh, P1*P1)
q = Function(ME)
qa = q.vector().get_local()
B = 0.64*1e4
C = 0.35*1e4
s0 = B/C/2
qa[::2] = s0*np.cos(2*anglea)
qa[1::2] = s0*np.sin(2*anglea)
q.vector().set_local(qa)
output_file = HDF5File(mesh.mpi_comm(), './image/q_'+name+'.h5', "w")
output_file.write(q, "solution")
output_file.close()
    
