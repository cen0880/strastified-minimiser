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
N = 4
ax = [0] * N
ay = [0] * N
d = [0] * N
theta = [0] * N

# sTRI
d = [-1.0,1.0,1.0,1.0]

# boundary defects
K = 6
c = [0] * K
g = [0] * K
c = [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]
#c = [2.0,2.0,2.0,2.0,2.0,2.0]
#c = [2.0,2.0,-1.0,-1.0,-1.0,-1.0]
#c = [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]
#c = [2.0,-1.0,2.0,-1.0,2.0,-1.0]
g[0] = -2.0*np.pi/3.0
for i in range(K-1):
    g[i+1] = g[i] - 2.0*c[i+1]*np.pi/3.0

class MyExpression0(UserExpression):
   def eval(self, value, x):
      value[0] = 0
      for i in range(N):
         value[0] = value[0] - d[i]*(np.arctan2(sin(theta[i])*(x[0]-ax[i]) + cos(theta[i])*(x[1]-ay[i]),cos(theta[i])*(x[0]-ax[i])-sin(theta[i])*(x[1]-ay[i])) - theta[i])
class MyExpression01(UserExpression):
   def eval(self, value, x):
      value[0] = 1
      for i in range(N):
         if (x[0]-ax[i])**2 + (x[1]-ay[i])**2 < 1e-4:
            value[0] = 0

for i in range(1,10):
      ax = [0,np.cos(0),np.cos(2.0*np.pi/3.0),np.cos(2.0*np.pi/3.0)]
      ax[:] = [x * i*0.1 for x in ax]
      ay = [0,np.sin(0),np.sin(2.0*np.pi/3.0),-np.sin(2.0*np.pi/3.0)]
      ay[:] = [x * i*0.1 for x in ay]
      for j in range(N):
      theta[j] = np.pi-np.arctan2(-ay[j],1.0-ax[j])
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

      anglea = (phi.vector().get_local()-phi0.vector().get_local())/2.0
      D = VectorFunctionSpace(mesh,'Lagrange',1,dim=2)
      q = Function(D)
      q11, q12 = split(q)
      qa = q.vector().get_local()
      qa[::2] = np.cos(2.0*anglea)
      qa[1::2] = np.sin(2.0*anglea)
      q.vector().set_local(qa)
      phi01 = interpolate(MyExpression01(),V)
      W = assemble(inner(grad(q11),grad(q11))*phi01*dx+inner(grad(q12),grad(q12))*phi01*dx)
      print(i,W)
