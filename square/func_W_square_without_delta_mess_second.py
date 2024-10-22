from mshr import *
import numpy as np
from dolfin import *
import copy
# parameters["form_compiler"]["optimize"] = True
# parameters["form_compiler"]["cpp_optimize"] = True
#set_log_active(False)
#set_log_level(40)
#import logging
#ffc_logger = logging.getLogger('FFC')
#ffc_logger.setLevel(logging.WARNING)


K = 4 # number of domain corners
bx = [1.0, 0.0, -1.0, 0.0] # location x of domain corners
by = [0.0, 1.0, 0.0, -1.0] # location y of domain corners
N = 3 # number of interior defects
d = np.array([-1.0,1.0,-1.0]) # winding number of interior defects
c = np.array([1/2,-1/2,-1/2,-1/2])  # winding number of corner defects

ll = 1e-2
bc = []
Hlimxa = [0]*N
Hlimya = [0]*N
Hlimxan = [0]*N
Hlimyan = [0]*N
Hlimxb = [0]*K
Hlimyb = [0]*K
Hlimxbn = [0]*K
Hlimybn = [0]*K
# Create mesh and build function space
domain_vertices = [Point(1.0,0.0),
        Point(0.0,1.0),
        Point(-1.0,0.0),
        Point(0.0,-1.0),
        Point(1.0,0.0)]
# Generate mesh and plot
domain = Polygon(domain_vertices)
mesh = generate_mesh(domain, 64)
for _ in range(3):
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    for cc in cells(mesh):
        if cc.midpoint().distance(Point(0, 0)) > np.sqrt(2.0) / 2.0 - 0.01:
            cell_markers[cc] = True
        else:
            cell_markers[cc] = False
    # Refine mesh
    mesh = refine(mesh, cell_markers)

parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
ME = FunctionSpace(mesh, 'Lagrange', 1)
HH = Function(ME)
v = TestFunction(ME)
n = FacetNormal(mesh)
bm = MeshFunction('size_t', mesh,1)
bm.set_all(0)

# boundary
class BC1(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 0.0 and x[1] > 0.0 and on_boundary
bx1 = BC1()
bx1.mark(bm, 1)
class BC2(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < -0.0 and x[1] > 0.0 and on_boundary
bx2 = BC2()
bx2.mark(bm, 2)
class BC3(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < -0.0 and x[1] < 0.0 and on_boundary
bx3 = BC3()
bx3.mark(bm, 3)
class BC4(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 0.0 and x[1] < 0.0 and on_boundary
bx4 = BC4()
bx4.mark(bm, 4)

ds = Measure('ds', domain=mesh, subdomain_data=bm) 
# calculate g, ga and gb are the first and second part of g
Ex = Expression(
("di*(x[0]-axi)/((x[0]-axi)*(x[0]-axi)+(x[1]-ayi)*(x[1]-ayi))","di*(x[1]-ayi)/((x[0]-axi)*(x[0]-axi)+(x[1]-ayi)*(x[1]-ayi))"),
degree=1,
di=1,
axi=0,
ayi=0)
ga = [Ex]*N
gb = [Ex]*K
for j in range(N):
    ga[j] = Expression(
    ("di*(x[0]-axi)/((x[0]-axi)*(x[0]-axi)+(x[1]-ayi)*(x[1]-ayi))","di*(x[1]-ayi)/((x[0]-axi)*(x[0]-axi)+(x[1]-ayi)*(x[1]-ayi))"),
    degree=1,
    di=d[j],
    axi=0,
    ayi=0)
for j in range(K):
    gb[j] = Expression(
    ("2*K/(K-2)*ci*(x[0]-bxi)/((x[0]-bxi)*(x[0]-bxi)+(x[1]-byi)*(x[1]-byi))","2*K/(K-2)*ci*(x[1]-byi)/((x[0]-bxi)*(x[0]-bxi)+(x[1]-byi)*(x[1]-byi))"),
    degree=1,
    ci=c[j],
    bxi=bx[j],
    byi=by[j],
    K = K)

error0 = Constant(0.0)
FF = inner(grad(HH), grad(v))*dx - (dot(gb[0],n))*v*(ds(2)+ds(3)) - (dot(gb[1],n))*v*(ds(3)+ds(4)) - (dot(gb[2],n))*v*(ds(1)+ds(4)) - (dot(gb[3],n))*v*(ds(1)+ds(2)) + error0/4.0/np.sqrt(2)*v*ds
for j in range(N):
    FF = FF - (-dot(ga[j],n))*v*ds
# errorb is error from gb part
errorb = assemble(dot(gb[0],n)*(ds(2)+ds(3))) + assemble(dot(gb[1],n)*(ds(3)+ds(4))) + assemble(dot(gb[2],n)*(ds(1)+ds(4))) + assemble(dot(gb[3],n)*(ds(1)+ds(2)))

# calculate gradient of partial W/partial ax, partial W/partial ay
def gradF(xx):
    ax = np.array(xx[:N])
    ay = np.array(xx[N:])
    Fx_tmp = np.zeros(N)
    Fy_tmp = np.zeros(N)

    error = errorb
    for j in range(N):
        ga[j].axi = ax[j]
        ga[j].ayi = ay[j]

    for k in range(N):
        ga[k].axi = ax[k] + ll
        error = errorb 
        for j in range(N):
            error = error + assemble(-dot(ga[j],n)*ds)
        error0.assign(error)
        a = derivative(FF, HH)
        problem = NonlinearVariationalProblem(FF, HH, bc, a)
        solver = NonlinearVariationalSolver(problem)
        
        prm = solver.parameters
        prm['newton_solver']['linear_solver'] = "lu"
        prm['newton_solver']['absolute_tolerance'] = 1E-4
        prm['newton_solver']['relative_tolerance'] = 1E-7
        prm['newton_solver']['maximum_iterations'] = 10
        prm['newton_solver']['relaxation_parameter'] = 1.0
        solver.solve()
        c0 = assemble(HH*ds)/4.0/np.sqrt(2)
        for j in range(K):
            Hlimxb[j] = HH(bx[j],by[j]) - c0
        for j in range(N):
            if j==k:
                Hlimxa[j] = HH(ax[j]+ll,ay[j]) - c0
            else:
                Hlimxa[j] = HH(ax[j],ay[j]) - c0
        ga[k].axi = ax[k]


        ga[k].ayi = ay[k]+ll
        error = errorb 
        for j in range(N):
            error = error + assemble(-dot(ga[j],n)*ds)
        error0.assign(error)
        
        a = derivative(FF, HH)
        problem = NonlinearVariationalProblem(FF, HH, bc, a)
        solver = NonlinearVariationalSolver(problem)
        prm = solver.parameters
        prm['newton_solver']['linear_solver'] = "lu"
        prm['newton_solver']['absolute_tolerance'] = 1E-4
        prm['newton_solver']['relative_tolerance'] = 1E-7
        prm['newton_solver']['maximum_iterations'] = 10
        prm['newton_solver']['relaxation_parameter'] = 1.0
        solver.solve()
        c0 = assemble(HH*ds)/4.0/np.sqrt(2)
        for j in range(K):
            Hlimyb[j] = HH(bx[j],by[j]) - c0
        for j in range(N):
            if j==k:
                Hlimya[j] = HH(ax[j],ay[j]+ll) - c0
            else:
                Hlimya[j] = HH(ax[j],ay[j]) - c0
        ga[k].ayi = ay[k]

        ga[k].axi = ax[k] - ll
        error = errorb 
        for j in range(N):
            error = error + assemble(-dot(ga[j],n)*ds)
        error0.assign(error)
        a = derivative(FF, HH)
        problem = NonlinearVariationalProblem(FF, HH, bc, a)
        solver = NonlinearVariationalSolver(problem)
        
        prm = solver.parameters
        prm['newton_solver']['linear_solver'] = "lu"
        prm['newton_solver']['absolute_tolerance'] = 1E-4
        prm['newton_solver']['relative_tolerance'] = 1E-7
        prm['newton_solver']['maximum_iterations'] = 10
        prm['newton_solver']['relaxation_parameter'] = 1.0
        solver.solve()
        c0 = assemble(HH*ds)/4.0/np.sqrt(2)
        for j in range(K):
            Hlimxbn[j] = HH(bx[j],by[j]) - c0
        for j in range(N):
            if j==k:
                Hlimxan[j] = HH(ax[j]-ll,ay[j]) - c0
            else:
                Hlimxan[j] = HH(ax[j],ay[j]) - c0
        ga[k].axi = ax[k]


        ga[k].ayi = ay[k]-ll
        error = errorb 
        for j in range(N):
            error = error + assemble(-dot(ga[j],n)*ds)
        error0.assign(error)
        
        a = derivative(FF, HH)
        problem = NonlinearVariationalProblem(FF, HH, bc, a)
        solver = NonlinearVariationalSolver(problem)
        prm = solver.parameters
        prm['newton_solver']['linear_solver'] = "lu"
        prm['newton_solver']['absolute_tolerance'] = 1E-4
        prm['newton_solver']['relative_tolerance'] = 1E-7
        prm['newton_solver']['maximum_iterations'] = 10
        prm['newton_solver']['relaxation_parameter'] = 1.0
        solver.solve()
        c0 = assemble(HH*ds)/4.0/np.sqrt(2)
        for j in range(K):
            Hlimybn[j] = HH(bx[j],by[j]) - c0
        for j in range(N):
            if j==k:
                Hlimyan[j] = HH(ax[j],ay[j]-ll) - c0
            else:
                Hlimyan[j] = HH(ax[j],ay[j]) - c0
        ga[k].ayi = ay[k]

        for j in range(K):
            Fx_tmp[k] = Fx_tmp[k] - (2.0*K/(K-2)+1)*np.pi * d[k] * c[j] * (ax[k] - bx[j]) / ((ax[k] - bx[j])**2 + (
                ay[k] - by[j])**2) - np.pi * c[j] * (Hlimxb[j] - Hlimxbn[j]) / ll/2.0
            Fy_tmp[k] = Fy_tmp[k] - (2.0*K/(K-2)+1)*np.pi * d[k] * c[j] * (ay[k] - by[j]) / ((ax[k] - bx[j])**2 + (
                ay[k] - by[j])**2) - np.pi * c[j] * (Hlimyb[j] - Hlimybn[j]) / ll/2.0
        for j in range(N):
            if k != j:
                Fx_tmp[k] = Fx_tmp[k] + 2.0 * np.pi * d[k] * d[j] * \
                   (ax[k] - ax[j]) / ((ax[k] - ax[j])**2 + (ay[k] - ay[j])**2)
                Fy_tmp[k] = Fy_tmp[k] + 2.0 * np.pi * d[k] * d[j] * \
                    (ay[k] - ay[j]) / ((ax[k] - ax[j])**2 + (ay[k] - ay[j])**2)
        for j in range(N):
                Fx_tmp[k] = Fx_tmp[k] + np.pi * d[j] * (Hlimxa[j] - Hlimxan[j]) / ll/2.0
                Fy_tmp[k] = Fy_tmp[k] + np.pi * d[j] * (Hlimya[j] - Hlimyan[j]) / ll/2.0

    return np.append(Fx_tmp, Fy_tmp)

def H(xx, vv, l):
    ret = - (gradF(xx + l * vv) - gradF(xx - l * vv)) / (2 * l)
    return ret

class MyScalarExpression(UserExpression):
    def __init__(self, *args):
        UserExpression.__init__(self)

    def eval(self, value, x):
        pass

    def value_shape(self):
        return ()

def draw(filename, xx, V):
    ax = np.array(xx[:N])
    ay = np.array(xx[N:])

    phi = Function(ME)
    vphi = TestFunction(ME)

    gg = np.empty(K)
    gg[0] = -np.pi/2.0
    for i in range(K - 1):
        gg[i + 1] = gg[i] + 2.0 * c[i + 1] * np.pi

    theta = np.pi - np.arctan2(-ay, 1 - ax)

    class MyExpressionphi0(MyScalarExpression):
        def eval(self, value, x):
            val = - d * (np.arctan2(np.sin(theta) * (x[0] - ax) + np.cos(theta) * (x[1] - ay),
                    np.cos(theta) * (x[0] - ax) - np.sin(theta) * (x[1] - ay)) - theta)
            value[0] = np.sum(val)
    phi0 = interpolate(MyExpressionphi0(), ME)

    def C1(x, on_boundary):
        return x[0] > 0.0 and x[1] > 0.0 and on_boundary
    bc1 = DirichletBC(ME, phi0 + gg[0], C1)

    def C2(x, on_boundary):
        return x[0] < 0.0 and x[1] > 0.0 and on_boundary
    bc2 = DirichletBC(ME, phi0 + gg[1], C2)

    def C3(x, on_boundary):
        return x[0] < 0.0 and x[1] < 0.0 and on_boundary
    bc3 = DirichletBC(ME, phi0 + gg[2], C3)

    def C4(x, on_boundary):
        return x[0] > 0.0 and x[1] < 0.0 and on_boundary
    bc4 = DirichletBC(ME, phi0 + gg[3], C4)

    bcphi = [bc1, bc2, bc3, bc4]

    FF = inner(grad(phi), grad(vphi)) * dx
    a = derivative(FF, phi)
    problem = NonlinearVariationalProblem(FF, phi, bcphi, a)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm['newton_solver']['linear_solver'] = "lu"
    prm['newton_solver']['absolute_tolerance'] = 1E-4
    prm['newton_solver']['relative_tolerance'] = 1E-7
    prm['newton_solver']['maximum_iterations'] = 10
    prm['newton_solver']['relaxation_parameter'] = 1.0
    solver.solve()

    # output a .vtu file for nematic direction visualization in Paraview
    anglea = (phi.vector().get_local() - phi0.vector().get_local()) / 2.0
    D = VectorFunctionSpace(mesh, 'Lagrange', 1, dim=2)
    dd = Function(D)
    da = dd.vector().get_local()
    da[::2] = np.cos(anglea)
    da[1::2] = np.sin(anglea)
    dd.vector().set_local(da)
    vtkfile = File(filename + '.pvd')
    vtkfile << dd

    np.savez(filename + '.npz', xx=xx, V=V)
    
def innp(xx, yy):
    return np.dot(xx.T, yy)
    # return xx.dot(yy)


def readfile(filename):
    data = np.load(filename + '.npz')
    return data
