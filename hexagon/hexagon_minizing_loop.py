import sys      

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

N = 3
ax = [0] * N
ay = [0] * N
ax_tmp = [0] * N
ay_tmp = [0] * N
d = [0] * N
d = [1.0,-1.0,-1.0]
ax = [0.0, 1.0/2.0, 0.0]
ay = [0.0, 1.0/2.0/sqrt(3.0), 1.0/sqrt(3.0)]
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
K = 6
bx = [0] * N
by = [0] * N
c = [0] * K
bx = [1.0,0.5,-0.5,-1.0,-0.5,0.5]
by = [0.0,0.5*sqrt(3.0),0.5*sqrt(3.0),0.0,-0.5*sqrt(3.0),-0.5*sqrt(3.0)]
#c = np.multiply([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0],-1.0/3.0)
c = [-2.0/3.0,-2.0/3.0,-2.0/3.0,1.0/3.0,1.0/3.0,1.0/3.0]

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
while (t<T):
    limx = [0]*N
    limy = [0]*N
    g = interpolate(MyExpressiong(),V)
    error = assemble((delta-dot(grad(g),n))*ds)
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
    H.vector().set_local(H.vector().get_local() - assemble(H*dx)/(3.0*sqrt(3.0)/2.0))
    W = 0
    for ki in range(N):
        for kj in range(N):
            if ki!=kj:
               W = W - np.pi*d[ki]*d[kj]*np.log(sqrt((ax[ki]-ax[kj])**2+(ay[ki]-ay[kj])**2))
    for ki in range(K):
        for kj in range(N):
            W = W + np.pi*c[ki]*d[kj]*np.log(sqrt((bx[ki]-ax[kj])**2+(by[ki]-ay[kj])**2))
               #print(d[ki]*d[kj],np.log(sqrt((ax[ki]-ax[kj])**2+(ay[ki]-ay[kj])**2)),W)
      
    for ki in range(N):
        W = W - np.pi*d[ki]*H(ax[ki],ay[ki])
    for ki in range(K):
        W = W + np.pi*c[ki]*H(bx[ki],by[ki])
    #  W = W + assemble((H+g)*delta*ds)/2.0
    print(ax,ay,W)
    for k in range(N):
        limx = [0]*N
        limy = [0]*N
        limx[k] = l
        glimx = interpolate(MyExpressiong(),V)
        error = assemble((delta-dot(grad(glimx),n))*ds)
        F = inner(grad(Hlimx), grad(v))*dx - (delta-dot(grad(glimx),n)-error/6.0)*v*ds
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
        Hlimx.vector().set_local(Hlimx.vector().get_local() - assemble(Hlimx*dx)/(3.0*sqrt(3.0)/2.0))

        limx = [0]*N
        limy = [0]*N
        limy[k] = l
        glimy = interpolate(MyExpressiong(),V)
        error = assemble((delta-dot(grad(glimy),n))*ds)
        F = inner(grad(Hlimy), grad(v))*dx - (delta-dot(grad(glimy),n)-error/6.0)*v*ds
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
        Hlimy.vector().set_local(Hlimy.vector().get_local() - assemble(Hlimy*dx)/(3.0*sqrt(3.0)/2.0))

        for j in range(K):
                ax_tmp[k] = ax_tmp[k] - dt*np.pi*d[k]*c[j]*(ax[k]-bx[j])/((ax[k]-bx[j])**2+(ay[k]-by[j])**2) - dt* np.pi*c[j]*(Hlimx(bx[j],by[j])-H(bx[j],by[j]))/l  
                ay_tmp[k] = ay_tmp[k] - dt*np.pi*d[k]*c[j]*(ay[k]-by[j])/((ax[k]-bx[j])**2+(ay[k]-by[j])**2) - dt* np.pi*c[j]*(Hlimy(bx[j],by[j])-H(bx[j],by[j]))/l  
                #print("k=",k,"j=",j, dt*np.pi*d[k]*c[j]*(ax[k]-bx[j])/((ax[k]-bx[j])**2+(ay[k]-by[j])**2),dt*np.pi*d[k]*c[j]*(ay[k]-by[j])/((ax[k]-bx[j])**2+(ay[k]-by[j])**2))

                #print("k=",k,"j=",j, Hlimx(bx[j],by[j])-H(bx[j],by[j]), Hx(bx[j],by[j]), Hlimy(bx[j],by[j])-H(bx[j],by[j]), Hy(bx[j],by[j]))
        for j in range(N):
                if k!=j:
                        ax_tmp[k] = ax_tmp[k] + dt*2.0*np.pi*d[k]*d[j]*(ax[k]-ax[j])/((ax[k]-ax[j])**2+(ay[k]-ay[j])**2)
                        ay_tmp[k] = ay_tmp[k] + dt*2.0*np.pi*d[k]*d[j]*(ay[k]-ay[j])/((ax[k]-ax[j])**2+(ay[k]-ay[j])**2)
                        #print("k=",k,"j=",j, dt*np.pi*d[k]*d[j]*(ax[k]-ax[j])/((ax[k]-ax[j])**2+(ay[k]-ay[j])**2),dt*np.pi*d[k]*d[j]*(ay[k]-ay[j])/((ax[k]-ax[j])**2+(ay[k]-ay[j])**2))

                #print("k=",k,"j=",j, Hlimx(ax[j],ay[j])-H(ax[j],ay[j]), Hx(ax[j],ay[j]), Hlimy(ax[j],ay[j])-H(ax[j],ay[j]), Hy(ax[j],ay[j]))
        for j in range(N):
                if k!=j:
                        ax_tmp[k] = ax_tmp[k] + dt* np.pi*d[j]*(Hlimx(ax[j],ay[j])-H(ax[j],ay[j]))/l 
                        ay_tmp[k] = ay_tmp[k] + dt* np.pi*d[j]*(Hlimy(ax[j],ay[j])-H(ax[j],ay[j]))/l

        ax_tmp[k] = ax_tmp[k] + dt* np.pi*d[k]*(Hlimx(ax[k]+l,ay[k])-H(ax[k],ay[k]))/l
        ay_tmp[k] = ay_tmp[k] + dt* np.pi*d[k]*(Hlimy(ax[k],ay[k]+l)-H(ax[k],ay[k]))/l

    ax = ax_tmp.copy()
    ay = ay_tmp.copy()

    t = t + dt;       
    print(t, ax, ay, file=sys.stderr)
