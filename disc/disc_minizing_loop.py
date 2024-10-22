import sys      

from dolfin import *
from mshr import *
import numpy as np


mesh = generate_mesh(Circle(Point(0,0),1),256)
for i in range(2):
#  for j in range(K):
    # Mark cells for refinement
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    for cc in cells(mesh):
        if cc.midpoint().distance(Point(0,0)) > 1.0-0.02:
            cell_markers[cc] = True
        else:
            cell_markers[cc] = False
    # Refine mesh
    mesh = refine(mesh, cell_markers)
# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
# Create mesh and build function space
V = FunctionSpace(mesh, 'Lagrange', 1)
# Define the test and trial functions
H = Function(V)
Hlimx = Function(V)
Hlimy = Function(V)
v = TestFunction(V)
n = FacetNormal(mesh)
bc = []
dt = 1e-3
eps = 1e-2
T = 1.0
t = 0.0
l = 1e-3

N = 1
ax = [0] * N
ay = [0] * N
ax_tmp = [0] * N
ay_tmp = [0] * N
d = [0] * N
d = [1.0,-1.0]
ax = [1.0/2.0, -1.0/2.0]
ay = [0.0, 0.0]
#d = [1.0,1.0,1.0,-1.0]
#ax = [0.25*sqrt(3.0),-0.25*sqrt(3.0)]
#ay = [0.25,-0.25]
#ax = [0.5,-0.25,-0.25,0.0]
#ay = [0.0,0.25*sqrt(3.0),-0.25*sqrt(3.0),0.0]
#ax = np.multiply([0.5,-0.25,-0.25,0.0],0.2)
#ay = np.multiply([0.0,0.25*sqrt(3.0),-0.25*sqrt(3.0),0.0],0.2)
ax_tmp = ax.copy()
ay_tmp = ay.copy()
limx = [0]*N
limy = [0]*N

bc = []
class MyExpressiong(UserExpression):
    def __init__(self,*args):
        UserExpression.__init__(self)
    def eval(self, value, x):
        value[0] = 0
        #print(limx)
        for i in range(N):
            value[0] = value[0] + d[i]*np.log(sqrt((x[0]-ax[i]-limx[i])**2 + (x[1]-ay[i]-limy[i])**2))
            #if (sqrt((x[0]-ax[i]-limx[i])**2 + (x[1]-ay[i]-limy[i])**2))==0:
            #    print(i,x[0],ax[i],x[1],ay[i])
    def value_shape(self):
        return()
class MyExpressiondelta(UserExpression):
    def __init__(self,*args):
        UserExpression.__init__(self)
    def eval(self, value, x):
        value[0] = 2.0
    def value_shape(self):
        return()
delta = interpolate(MyExpressiondelta(),V)
while (t<T):
    limx = [0]*N
    limy = [0]*N
    g = interpolate(MyExpressiong(),V)
    error = assemble((delta-dot(grad(g),n))*ds)
    F = inner(grad(H), grad(v))*dx - (delta-dot(grad(g),n)-error/2.0/np.pi)*v*ds 
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
    H.vector().set_local(H.vector().get_local() - assemble(H*ds))
    W = 0
    for ki in range(N):
        for kj in range(N):
            if ki!=kj:
               W = W - np.pi*d[ki]*d[kj]*np.log(sqrt((ax[ki]-ax[kj])**2+(ay[ki]-ay[kj])**2))
      
    for ki in range(N):
        W = W - np.pi*d[ki]*H(ax[ki],ay[ki])
    W = W + assemble(g*ds) + np.pi*2.0*H(0.0,0.0)
    print(ax,ay,W)
    for k in range(N):
        limx = [0]*N
        limy = [0]*N
        limx[k] = l
        glimx = interpolate(MyExpressiong(),V)
        error = assemble((delta-dot(grad(glimx),n))*ds)
        F = inner(grad(Hlimx), grad(v))*dx - (delta-dot(grad(glimx),n)-error/2.0/np.pi)*v*ds
        a = derivative(F, Hlimx)
        problem = NonlinearVariationalProblem(F, Hlimx, bc, a)
        solver  = NonlinearVariationalSolver(problem)
        prm = solver.parameters
        prm['newton_solver']['linear_solver'] = "lu"
        prm['newton_solver']['absolute_tolerance'] = 1E-4
        prm['newton_solver']['relative_tolerance'] = 1E-7
        prm['newton_solver']['maximum_iterations'] = 10
        prm['newton_solver']['relaxation_parameter'] = 1.0
        solver.solve()
        Hlimx.vector().set_local(Hlimx.vector().get_local() - assemble(Hlimx*ds))

        limx = [0]*N
        limy = [0]*N
        limy[k] = l
        glimy = interpolate(MyExpressiong(),V)
        error = assemble((delta-dot(grad(glimy),n))*ds)
        F = inner(grad(Hlimy), grad(v))*dx - (delta-dot(grad(glimy),n)-error/2.0/np.pi)*v*ds
        a = derivative(F, Hlimy)
        problem = NonlinearVariationalProblem(F, Hlimy, bc, a)
        solver  = NonlinearVariationalSolver(problem)
        prm = solver.parameters
        prm['newton_solver']['linear_solver'] = "lu"
        prm['newton_solver']['absolute_tolerance'] = 1E-4
        prm['newton_solver']['relative_tolerance'] = 1E-7
        prm['newton_solver']['maximum_iterations'] = 10
        prm['newton_solver']['relaxation_parameter'] = 1.0
        solver.solve()
        Hlimy.vector().set_local(Hlimy.vector().get_local() - assemble(Hlimy*ds))

       for j in range(N):
                if k!=j:
                        ax_tmp[k] = ax_tmp[k] + dt*2.0*np.pi*d[k]*d[j]*(ax[k]-ax[j])/((ax[k]-ax[j])**2+(ay[k]-ay[j])**2)
                        ay_tmp[k] = ay_tmp[k] + dt*2.0*np.pi*d[k]*d[j]*(ay[k]-ay[j])/((ax[k]-ax[j])**2+(ay[k]-ay[j])**2)
        for j in range(N):
                if k!=j:
                        ax_tmp[k] = ax_tmp[k] + dt* np.pi*d[j]*(Hlimx(ax[j],ay[j])-H(ax[j],ay[j]))/l 
                        ay_tmp[k] = ay_tmp[k] + dt* np.pi*d[j]*(Hlimy(ax[j],ay[j])-H(ax[j],ay[j]))/l
        ax_tmp[k] = ax_tmp[k] + dt* np.pi*d[k]*(Hlimx(ax[k]+l,ay[k])-H(ax[k],ay[k]))/l - dt*assemble(d[k]*(ax[k]-x[0])/((ax[k]-x[0])**2+(ay[k]-x[1])**2)*ds)-dt*2.0*np.pi*(Hlimx(0.0,0.0)-H(0.0,0.0))/l
        ay_tmp[k] = ay_tmp[k] + dt* np.pi*d[k]*(Hlimy(ax[k],ay[k]+l)-H(ax[k],ay[k]))/l - dt*assemble(d[k]*(ay[k]-x[1])/((ax[k]-x[0])**2+(ay[k]-x[1])**2)*ds)-dt*2.0*np.pi*(Hlimy(0.0,0.0)-H(0.0,0.0))/l


    ax = ax_tmp.copy()
    ay = ay_tmp.copy()

    t = t + dt;       
    print(t, ax, ay, file=sys.stderr)
