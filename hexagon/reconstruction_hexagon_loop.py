from dolfin import *
from mshr import *
import numpy as np
domain_vertices = [Point(1.0,0.0),
        Point(0.5,0.5*sqrt(3.0)),
        Point(-0.5,0.5*sqrt(3.0)),
        Point(-1.0,0.0),
        Point(-0.5,-0.5*sqrt(3.0)),
        Point(0.5,-0.5*sqrt(3.0)),
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
# interior defects
N = 7
ax = [0] * N
ay = [0] * N
d = [0] * N
theta = [0] * N
# BD
#d = [1.0,1.0]
#ax = [0.55*0.5*sqrt(3.0),-0.55*0.5*sqrt(3.0)]
#ay = [0.55*0.5,-0.55*0.5]
#c = [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]
# sTRI
#d = [1.0,1.0,1.0,-1.0]
#ax = [0.5,-0.25,-0.25,0.0]
#ay = [0.0,0.25*sqrt(3.0),-0.25*sqrt(3.0),0.0]
#c = [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]
# TRI
#d = [-1.0]
#ax = [0.0]
#ay = [0.0]
# H
d = [2.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]
ax = [0,np.cos(np.pi/6.0),np.cos(np.pi/2.0),np.cos(5.0*np.pi/6.0),np.cos(7.0*np.pi/6.0),np.cos(3.0*np.pi/2.0),np.cos(11.0*np.pi/6.0)]
ax[:] = [0.66*x for x in ax]
ay = [0,np.sin(np.pi/6.0),np.sin(np.pi/2.0),np.sin(5.0*np.pi/6.0),np.sin(7.0*np.pi/6.0),np.sin(3.0*np.pi/2.0),np.sin(11.0*np.pi/6.0)]
ay[:] = [0.66*y for y in ay]
      
# boundary defects
K = 6
c = [0] * K
g = [0] * K
#c = [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]
c = [2.0,2.0,2.0,2.0,2.0,2.0]
#c = [2.0,2.0,-1.0,-1.0,-1.0,-1.0]
#c = [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]
#c = [2.0,-1.0,2.0,-1.0,2.0,-1.0]
g[0] = -2.0*np.pi/3.0
for i in range(K-1):
    g[i+1] = g[i] - 2.0*c[i+1]*np.pi/3.0
for i in range(N):
    theta[i] = np.pi-np.arctan2(-ay[i],1.0-ax[i])

class MyExpression0(UserExpression):
    def eval(self, value, x):
        value[0] = 0
        for i in range(N):
            value[0] = value[0] - d[i]*(np.arctan2(sin(theta[i])*(x[0]-ax[i]) + cos(theta[i])*(x[1]-ay[i]),cos(theta[i])*(x[0]-ax[i])-sin(theta[i])*(x[1]-ay[i])) - theta[i])

phi0 = interpolate(MyExpression0(),V)

def C1(x, on_boundary):
    return x[0] > 0.5 and x[1] > 0.0 and on_boundary
bc1 = DirichletBC(V, phi0 + g[0], C1)
def C2(x, on_boundary):
    return x[0] > -0.5 and x[0] < 0.5 and x[1] > 0.0 and on_boundary
bc2 = DirichletBC(V, phi0 + g[1], C2)
def C3(x, on_boundary):
    return x[0] < -0.5 and x[1] > 0.0 and on_boundary
bc3 = DirichletBC(V, phi0 + g[2], C3)
def C4(x, on_boundary):
    return x[0] < -0.5 and x[1] < 0.0 and on_boundary
bc4 = DirichletBC(V, phi0 + g[3], C4)
def C5(x, on_boundary):
    return x[0] > -0.5 and x[0] < 0.5 and x[1] < 0.0 and on_boundary
bc5 = DirichletBC(V, phi0 + g[4], C5)
def C6(x, on_boundary):
    return x[0] > 0.5 and x[1] < 0.0 and on_boundary
bc6 = DirichletBC(V, phi0 + g[5], C6)  

bc = [bc1,bc2,bc3,bc4,bc5,bc6]

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
vtkfile = File('./angle_H.pvd')
vtkfile << angle
D = VectorFunctionSpace(mesh,'Lagrange',1,dim=2)
dd = Function(D)
da = dd.vector().get_local()
da[::2] = np.cos(anglea)
da[1::2] = np.sin(anglea)
dd.vector().set_local(da)
vtkfile = File('./direction_H.pvd')
vtkfile << dd

# output .h5 data file
output_file = HDF5File(mesh.mpi_comm(), "./image/angle_H.h5", "w")
output_file.write(angle, "solution")
output_file.close()
    
    
