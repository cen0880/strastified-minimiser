import numpy as np
import numpy.linalg as la
from func_with_boundary_disc import numindex
from func_with_boundary_disc import readfile
from func_with_boundary_disc import readfileV
from func_with_boundary_disc import F
from func_with_boundary_disc import H
from func_with_boundary_disc import innp
from func_with_boundary_disc import nor
from func_with_boundary_disc import draw
#parameter
dt = 0.0002
# maxiter=500
maxiter=500
# minl = 1e-6
# l = 1e-6
minl = 1e-1
l = 1e-1

filename = "./image/angle_TRI.h5"
nb = numindex()
x = readfile(filename)
k = 2
n = x.shape[0] - nb
N = n/2
#V = np.random.randn(n,k)
V = readfileV(k)
for i in np.arange(k):
    V[:,i:i+1] = V[:,i:i+1] - np.dot(V[:,0:i],innp(V[:,0:i], V[:,i:i+1]))
    V[:,i:i+1] = V[:,i:i+1]/nor(V[:,i:i+1])
alpha = np.zeros((k,1),dtype = float)
U = np.zeros((n,k),dtype = float)
W = np.zeros((n,k),dtype = float)
Y = np.zeros((n,k),dtype = float)
Vp = np.zeros((n,k),dtype = float)
Up = np.zeros((n,k),dtype = float)
# a = H(x,V[:,0:1],l)
# print 'a',a[0:k,0]
# print V[0:2,0:7]
# print F(x+np.ones((x.shape[0],1),dtype = np.float))
iter = 1
while iter<maxiter:
    emp = []
    for i in np.arange(k):
        U[:,i:i+1] = H(x, V[:,i:i+1], l)
        alpha[i] = innp(V[:,i:i+1], U[:,i:i+1])
        W[:,i:i+1] =   U[:,i:i+1] - alpha[i] * V[:,i:i+1]
        W[:,i:i+1] = W[:,i:i+1] - np.dot(V, innp(V, W[:,i:i+1]))
        W[:,i:i+1] = W[:,i:i+1] - np.dot(W[:,0:i], innp(W[:,0:i], W[:,i:i+1]))
        nrmW = nor(W[:,i:i+1])
        if nrmW > 1e-6:
            W[:,i:i+1] = W[:,i:i+1] / nrmW
            Y[:,i:i+1] = H (x, W[:,i:i+1], l)
        else:
            W[:,i:i+1] = np.zeros((n,1),dtype = float)
            Y[:,i:i+1] = np.zeros((n,1),dtype = float)
            emp.append(k+i)
    
    for i in np.arange(k):
        Vp[:,i:i+1] = Vp[:,i:i+1] - np.dot(np.concatenate((V,W,Vp[:,0:i]),axis=1),innp( np.concatenate((V,W,Vp[:,0:i]),axis=1), Vp[:,i:i+1]))
        nrmVpi = nor(Vp[:,i:i+1])
        if nrmVpi > 1e-6:
            Vp[:,i:i+1] = Vp[:,i:i+1] / nrmVpi
            Vp[:,i:i+1] = Vp[:,i:i+1] - np.dot(np.concatenate((V,W,Vp[:,0:i]),axis=1),innp(np.concatenate((V,W,Vp[:,0:i]),axis=1), Vp[:,i:i+1]))
            Vp[:,i:i+1] = Vp[:,i:i+1] / nor(Vp[:,i:i+1]);
            Up[:,i:i+1] = H (x, Vp[:,i:i+1], l);
        else:
            Vp[:,i:i+1] = np.zeros((n,1),dtype = float)
            Up[:,i:i+1] = np.zeros((n,1),dtype = float)
            emp.append(2*k+i)
    UU = np.concatenate((V,W,Vp),axis = 1)
    UU = np.delete(UU, emp, axis = 1)
    YY = np.concatenate((U,Y,Up),axis = 1)
    YY = np.delete(YY, emp, axis = 1)
    # print YY.shape
    Pn = innp(UU,YY)
    Pn = (Pn+Pn.T) / 2.0
    alpha, eta = la.eig(Pn)
    idx = alpha.argsort()[::1]
    alpha = alpha[idx]
    eta = eta[:,idx]
    Vp = np.copy(V)
    Up = np.copy(U)
    V = np.dot(UU,eta[:,0:k])
    alpha = alpha[0:k]
    print(alpha)
    if iter%100==0:
        print(iter)
    l = max(l/(1+dt),minl)
    iter = iter+1
for i in np.arange(k):
    draw(V[:,i:i+1],i+1)
