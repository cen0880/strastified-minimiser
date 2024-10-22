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
K = 6
bx = [0] * K
by = [0] * K
c = [0] * K
eps = 1e-2
bx = [1.0,0.5,-0.5,-1.0,-0.5,0.5]
by = [0.0,0.5*sqrt(3.0),0.5*sqrt(3.0),0.0,-0.5*sqrt(3.0),-0.5*sqrt(3.0)]

#c = np.multiply([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0],-1.0/3.0)
#defects_eps = 0.1
#defects = [Circle(Point(bx[j],by[j]),defects_eps) for j in range(K)]
#for (j, defect) in enumerate(defects):
#    domain.set_subdomain(j+1,defect)
mesh = generate_mesh(domain,64)
for i in range(2):
#  for j in range(K):
    # Mark cells for refinement
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    for cc in cells(mesh):
        if cc.midpoint().distance(Point(0,0)) > np.sqrt(3.0)/2.0-0.02:
            cell_markers[cc] = True
        else:
            cell_markers[cc] = False
    # Refine mesh
    mesh = refine(mesh, cell_markers)
#vtkfile = File('./image/mesh.pvd')
#vtkfile << mesh

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
#N = 4
N = 1
ax = [0] * N
ay = [0] * N
d = [0] * N
#d =[-1.0]
#c = [-2.0/3.0,1.0/3.0,-2.0/3.0,1.0/3.0,-2.0/3.0,1.0/3.0]
d = [-1.0]
c = [-2.0/3.0,1.0/3.0,-2.0/3.0,-2.0/3.0,1.0/3.0,1.0/3.0]
#c = [-2.0/3.0,-2.0/3.0,-2.0/3.0,1.0/3.0,1.0/3.0,1.0/3.0]
#d = [-1.0,1.0,1.0,1.0]
#d = [2,-1,-1,-1,-1,-1,-1]
#c = np.multiply([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0],2.0/3.0)
class MyExpressiong(UserExpression):
    def __init__(self,*args):
        UserExpression.__init__(self)
    def eval(self, value, x):
        value[0] = 0
        for i in range(N):
             value[0] = value[0] + d[i]*np.log(sqrt((x[0]-ax[i])**2+(x[1]-ay[i])**2))
    def value_shape(self):
        return()
class MyExpressiondelta0(UserExpression):
    def __init__(self,*args):
        UserExpression.__init__(self)
    def eval(self, value, x):
        value[0] = 0
        for i in range(K):
            value[0] = value[0] + 2.0*np.pi*c[i]*eps/((x[0]-bx[i])**2+(x[1]-by[i])**2+eps*eps)
    def value_shape(self):
        return()
class MyExpressiondelta(UserExpression):
    def __init__(self,*args):
        UserExpression.__init__(self)
    def eval(self, value, x):
        value[0] = 0
        for i in range(K):
            value[0] = value[0] + 2.0*np.pi*c[i]*eps/((x[0]-bx[i])**2+(x[1]-by[i])**2+eps*eps)/tmp
    def value_shape(self):
        return()
delta0 = interpolate(MyExpressiondelta0(),V)
if np.sum(c) == 0:
    delta = interpolate(MyExpressiondelta0(),V)
else:
    tmp = assemble(delta0*ds)/(np.sum(c)*2.0*np.pi)
    delta = interpolate(MyExpressiondelta(),V)
#vtkfile = File('./image/delta.pvd')
#vtkfile << delta
print("delta=", assemble(delta*ds),"2pi*sum(c) = ", np.sum(c)*2.0*np.pi)
file1 = open("neg_half_134.txt", "a+") 

for i in range(-10,20):   
#for i in [3,4,5]:
  for j in range(20):
    x = i*0.05
    y = j*0.05
    if (y<sqrt(3.0)*(x+1.0)) and (y<-sqrt(3.0)*(x-1.0)) and (y>=0.0) and (y<=sqrt(3.0)/2.0):
    #if (0.01*i*i + 0.01*j*j < 3.0/4.0):
      #ax = [0,np.cos(np.pi/6.0),np.cos(np.pi/2.0),np.cos(5.0*np.pi/6.0),np.cos(7.0*np.pi/6.0),np.cos(3.0*np.pi/2.0),np.cos(11.0*np.pi/6.0)]
      #ax[:] = [x * (0.6+i*0.01) for x in ax]
      #ay = [0,np.sin(np.pi/6.0),np.sin(np.pi/2.0),np.sin(5.0*np.pi/6.0),np.sin(7.0*np.pi/6.0),np.sin(3.0*np.pi/2.0),np.sin(11.0*np.pi/6.0)]
      #ay[:] = [y * (0.6+i*0.01) for y in ay]
      #ax = [0,np.cos(0),np.cos(2.0*np.pi/3.0),np.cos(2.0*np.pi/3.0)]
      #ax[:] = [x * i*0.1 for x in ax]
      #ay = [0,np.sin(0),np.sin(2.0*np.pi/3.0),-np.sin(2.0*np.pi/3.0)]
      #ay[:] = [x * i*0.1 for x in ay]
      #ax = [(0.5+i*0.01)*np.cos(np.pi/6.0),-(0.5+i*0.01)*np.cos(np.pi/6.0)]
      #ay = [(0.5+i*0.01)*np.sin(np.pi/6.0),-(0.5+i*0.01)*np.sin(np.pi/6.0)]
      ax[0] = x
      ay[0] = y
      g = interpolate(MyExpressiong(),V)
      #vtkfile = File('./image/g.pvd')
      #vtkfile << g
      error = assemble((delta-dot(grad(g),n))*ds)
      print(error)
      F = inner(grad(H), grad(v))*dx - (delta-dot(grad(g),n)-error/6.0)*v*ds
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
      #vtkfile = File('./image/H_0.pvd')
      #vtkfile << H
      W = 0
      for ki in range(N):
         for kj in range(N):
            if ki!=kj:
               W = W - np.pi*d[ki]*d[kj]*np.log(sqrt((ax[ki]-ax[kj])**2+(ay[ki]-ay[kj])**2))
      #for ki in range(K):
      #   for kj in range(N):
      #         W = W + np.pi*c[ki]*d[kj]*np.log(sqrt((bx[ki]-ax[kj])**2+(by[ki]-ay[kj])**2))
               #print(d[ki]*d[kj],np.log(sqrt((ax[ki]-ax[kj])**2+(ay[ki]-ay[kj])**2)),W)
      
      for ki in range(N):
         W = W - np.pi*d[ki]*H(ax[ki],ay[ki])
      #for ki in range(K):
      #   W = W + np.pi*c[ki]*H(bx[ki],by[ki])
      W = W + assemble((H+g)*delta*ds)/2.0
      print(i,ax,ay,W)
      file1.write(str(ax[0])+' '+str(ay[0])+' '+str(W)+'\n')
      #file1.write(str(ax[0])+' '+str(ax[1])+' '+str(W)+'\n')
file1.close() 
