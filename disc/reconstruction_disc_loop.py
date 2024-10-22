from dolfin import *
from mshr import *
import numpy as np
mesh = generate_mesh(Circle(Point(0,0),1),256)
# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
# Create mesh and build function space
V = FunctionSpace(mesh, 'Lagrange', 1)
# Define the test and trial functions
phi = Function(V)
v = TestFunction(V)
# interior defects
N = 4
ax = [0] * N
ay = [0] * N
d = [0] * N
theta = [0] * N
# Polar
d = [-1,1,1.0,1.0]
ax = [0,0.15*np.cos(0),0.15*np.cos(2.0*np.pi/3.0),0.15*np.cos(4.0*np.pi/3.0)]
ay = [0,0.15*np.sin(0),0.15*np.sin(2.0*np.pi/3.0),0.15*np.sin(4.0*np.pi/3.0)]
for i in range(N):
    theta[i] = np.pi-np.arctan2(-ay[i],1.0-ax[i])
# boundary defects
class MyExpression0(UserExpression):
    def eval(self, value, x):
        value[0] = 0
        for i in range(N):
            value[0] = value[0] - d[i]*(np.arctan2(sin(theta[i])*(x[0]-ax[i]) + cos(theta[i])*(x[1]-ay[i]),cos(theta[i])*(x[0]-ax[i])-sin(theta[i])*(x[1]-ay[i])) - theta[i])

phi0 = interpolate(MyExpression0(),V)
g0 = interpolate(Expression("2.0*(atan2(-x[1],-x[0])-pi)",degree=1),V)
def C(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, phi0 + g0, C)


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
vtkfile = File('./angle_TRI.pvd')
vtkfile << angle
D = VectorFunctionSpace(mesh,'Lagrange',1,dim=2)
dd = Function(D)
da = dd.vector().get_local()
da[::2] = np.cos(anglea)
da[1::2] = np.sin(anglea)
dd.vector().set_local(da)
vtkfile = File('./direction_TRI.pvd')
vtkfile << dd

# output .h5 data file
output_file = HDF5File(mesh.mpi_comm(), "./image/angle_TRI.h5", "w")
output_file.write(angle, "solution")
output_file.close()
    
