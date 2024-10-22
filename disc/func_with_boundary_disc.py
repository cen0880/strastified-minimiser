import numpy as np
from dolfin import *
from mshr import *
# import matplotlib
# matplotlib.use('PDF')
# import matplotlib.pyplot as plt

# domain_vertices = [Point(1.0, 0.0),
        # Point(0.5, sqrt(3)/2.0),
        # Point(-0.5, sqrt(3)/2.0),
        # Point(-1.0,0.0),
        # Point(-0.5, -sqrt(3)/2.0),
        # Point(0.5,- sqrt(3)/2.0),
        # Point(1.0, 0.0)]
# # Generate mesh and plot
# domain = Polygon(domain_vertices)
# triangle1 = Polygon([Point(0.0,0.0),Point(1.0,0.0),Point(3.0/4.0,sqrt(3)/4.0),Point(0.0,0.0)])
# triangle2 = Polygon([Point(0.0,0.0),Point(3.0/4.0,sqrt(3)/4.0),Point(0.5,sqrt(3)/2.0),Point(0.0,0.0)])
# triangle3 = Polygon([Point(0.0,0.0),Point(0.5,sqrt(3)/2.0),Point(0.0,sqrt(3)/2.0),Point(0.0,0.0)])
# triangle4 = Polygon([Point(0.0,0.0),Point(0.0,sqrt(3)/2.0),Point(-0.5,sqrt(3)/2.0),Point(0.0,0.0)])
# triangle5 = Polygon([Point(0.0,0.0),Point(-0.5,sqrt(3)/2.0),Point(-3.0/4.0,sqrt(3)/4.0),Point(0.0,0.0)])
# triangle6 = Polygon([Point(0.0,0.0),Point(-3.0/4.0,sqrt(3)/4.0),Point(-1.0,0.0),Point(0.0,0.0)])
# triangle7 = Polygon([Point(0.0,0.0),Point(-1.0,0.0),Point(-3.0/4.0,-sqrt(3)/4.0),Point(0.0,0.0)])
# triangle8 = Polygon([Point(0.0,0.0),Point(-3.0/4.0,-sqrt(3)/4.0),Point(-0.5,-sqrt(3)/2.0),Point(0.0,0.0)])
# triangle9 = Polygon([Point(0.0,0.0),Point(-0.5,-sqrt(3)/2.0),Point(0.0,-sqrt(3)/2.0),Point(0.0,0.0)])
# triangle10 = Polygon([Point(0.0,0.0),Point(-0.0,-sqrt(3)/2.0),Point(0.5,-sqrt(3)/2.0),Point(0.0,0.0)])
# triangle11 = Polygon([Point(0.0,0.0),Point(0.5,-sqrt(3)/2.0),Point(3.0/4.0,-sqrt(3)/4.0),Point(0.0,0.0)])
# triangle12 = Polygon([Point(0.0,0.0),Point(3.0/4.0,-sqrt(3)/4.0),Point(1.0,0.0),Point(0.0,0.0)])
# domain.set_subdomain(1,triangle1)
# domain.set_subdomain(2,triangle2)
# domain.set_subdomain(3,triangle3)
# domain.set_subdomain(4,triangle4)
# domain.set_subdomain(5,triangle5)
# domain.set_subdomain(6,triangle6)
# domain.set_subdomain(7,triangle7)
# domain.set_subdomain(8,triangle8)
# domain.set_subdomain(9,triangle9)
# domain.set_subdomain(10,triangle10)
# domain.set_subdomain(11,triangle11)
# domain.set_subdomain(12,triangle12)
# mesh = generate_mesh(domain, 64)
# mesh = RectangleMesh(Point(-1,-1),Point(1,1),64,64,"crossed")
#mesh = Mesh("./test_mesh_hexagon/test.xml")
mesh = generate_mesh(Circle(Point(0,0),1),256)
B = 0.64*1e4
C = 0.35*1e4
s2 = B*B/C/C/4.0
s0 = B/C/2.0
lambda_square=1000.0

parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = FunctionSpace(mesh, P1*P1)
m11, m12 = TestFunctions(ME)
q = Function(ME)
q11, q12 = split(q)

def boundary(x,on_boundary):
    return on_boundary
bc0 = DirichletBC(ME,(0.0,0.0),boundary)
bc1 = DirichletBC(ME,(1.0,1.0),boundary)
bc1.apply(q.vector())
qa = q.vector().get_local()
N0 = len(qa[::2])
index = []
for i in np.arange(2*N0):
    if qa[i]==1.0:
        index.append(i)
Cindex = np.delete(np.arange(2*N0),index)
tmpF = np.zeros((2*N0,1),dtype = float)
xx1 = np.zeros((2*N0,1),dtype = float)
xx2 = np.zeros((2*N0,1),dtype = float)
def numindex():
    return len(index)

def readfile(filename):
    V = FunctionSpace(mesh, 'Lagrange', 1)
    angle = Function(V)
    input_file = HDF5File(mesh.mpi_comm(), filename, "r")
    input_file.read(angle, "solution")
    input_file.close()
    qa = q.vector().get_local()
    qa[::2] = s0*np.cos(2*angle.vector().get_local())
    qa[1::2] = s0*np.sin(2*angle.vector().get_local())
    q.vector().set_local(qa)
    x = np.zeros((2*N0,1),dtype = float)
    x = qa.reshape(2*N0,1)
    return x.reshape(2*N0,1)
def readfileV(k):
    Vresult = np.zeros((2*N0,k),dtype = float)
    for i in np.arange(k):
        input_file = HDF5File(mesh.mpi_comm(), "./V_"+str(i+1)+".h5", "r")
        # input_file = HDF5File(mesh.mpi_comm(), "./image520/V"+str(i+1)+"_name.h5", "r")
        input_file.read(q, "solution")
        input_file.close()
        qa = q.vector().get_local()
        Vresult[:,i:i+1] = qa.reshape(2*N0,1)
    return np.delete(Vresult, index, axis = 0)

def F(x):
    qa = q.vector().get_local()
    qa = x.ravel()
    # qa[::2] = x[0:N0,0:1].ravel()
    # qa[1::2] = x[N0:2*N0,0:1].ravel()
    q.vector().set_local(qa)
    L = lambda_square*(q11*q11 + q12*q12 - s2)*q11*m11*dx + inner(grad(q11), grad(m11))*dx + lambda_square*(q11*q11 + q12*q12 - s2)*q12*m12*dx + inner(grad(q12), grad(m12))*dx
    b = assemble(L)
    bc0.apply(b)
    Ff = b.get_local()
    tmpF = Ff.reshape(2*N0,1)
    # tmpF[0:N0,0:1] = Ff[::2].reshape(N0,1)
    # tmpF[N0:2*N0,0:1] = Ff[1::2].reshape(N0,1)
    return -np.delete(tmpF, index, axis = 0)

def H(xx,vv,ll):
    # print xx1[Cindex,0].shape
    # print xx[Cindex,0].shape
    # print vv.shape
    xx1[index,0:1] = xx[index,0:1]
    xx1[Cindex,0:1] = xx[Cindex,0:1] + ll*vv
    xx2[index,0:1] = xx[index,0:1]
    xx2[Cindex,0:1] = xx[Cindex,0:1] - ll*vv
    tmp = - ( F (xx1) - F (xx2) ) / (2 * ll)
    tmp = tmp.reshape(len(tmp),1)
    return tmp

def innp(x,y):
    return np.dot(x.T,y)

def nor(x):
    return np.sqrt(np.dot(x.T,x))

def draw(x,i):
    xx1 = np.zeros((2*N0,1),dtype = float)
    xx1[Cindex,0:1] = x
    qa = xx1[0:N0*2,0].ravel()
    # qa[::2] = xx1[0:N0,0].ravel()
    # qa[1::2] = xx1[N0:2*N0,0].ravel()
    q.vector().set_local(qa)
    # plt.figure()
    # fig = plot(q.sub(0), title='v11')
    # plt.colorbar(fig)
    # plt.savefig('./image520/V11_RING'+str(i)+'.pdf')
    # plt.figure()
    # fig = plot(q.sub(1), title='v12')
    # plt.colorbar(fig)
    # plt.savefig('./image520/V12_RING'+str(i)+'.pdf')
    output_file = HDF5File(mesh.mpi_comm(), "./V_"+str(i)+".h5", "w")
    output_file.write(q, "solution")
    output_file.close()

