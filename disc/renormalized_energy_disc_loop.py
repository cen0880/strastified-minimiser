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
H = Function(V)
v = TestFunction(V)


n = FacetNormal(mesh)
bc = []
N = 4
ax = [0] * N
ay = [0] * N
d = [0] * N
d = [-1,1,1,1]
class MyExpression0(UserExpression):
   def eval(self, value, x):
      value[0] = 0
      for i in range(N):
         value[0] = value[0] + d[i]*np.log(sqrt((x[0]-ax[i])**2+(x[1]-ay[i])**2))
#file1 = open("TRI.txt", "w") 
for i in range(1,10):   
   #for j in range(i+1,10):
      ax = [0,np.cos(0),np.cos(2.0*np.pi/3.0),np.cos(2.0*np.pi/3.0)]
      ax[:] = [x * i*0.1 for x in ax]
      ay = [0,np.sin(0),np.sin(2.0*np.pi/3.0),-np.sin(2.0*np.pi/3.0)]
      ay[:] = [x * i*0.1 for x in ay]
      #ax = [0,0.1*i,0,-0.1*i,0]
      #ay = [0,0,0.1*i,0,-0.1*i]
      g0 = interpolate(MyExpression0(),V)
      error = assemble((2.0-dot(grad(g0),n))*ds)
      F = inner(grad(H), grad(v))*dx - (2.0-dot(grad(g0),n)-error/2.0/np.pi)*v*ds
      a = derivative(F, H)
      problem = NonlinearVariationalProblem(F, H, bc, a)
      solver  = NonlinearVariationalSolver(problem)
      prm = solver.parameters
      prm['newton_solver']['linear_solver'] = "lu"
      prm['newton_solver']['absolute_tolerance'] = 1E-4
      prm['newton_solver']['relative_tolerance'] = 1E-7
      prm['newton_solver']['maximum_iterations'] = 10
      prm['newton_solver']['relaxation_parameter'] = 1.0
      solver.solve()
      W = 0
      for ki in range(N):
         for kj in range(N):
            if ki!=kj:
               W = W -np.pi*d[ki]*d[kj]*np.log(sqrt((ax[ki]-ax[kj])**2+(ay[ki]-ay[kj])**2))
               #print(d[ki]*d[kj],np.log(sqrt((ax[ki]-ax[kj])**2+(ay[ki]-ay[kj])**2)),W)
      
      for ki in range(N):
         W = W -np.pi*d[ki]*H(ax[ki],ay[ki])
      W = W + assemble((g0)*ds) + 2.0*np.pi*H(0.0,0.0)
      print(assemble((g0)*ds))
      print(ax,ay,W)
      #file1.write(str(a1)+' '+str(a2)+' '+str(W)+'\n')
#file1.close() 
