import os
#os.environ['OPENBLAS_NUM_THREADS'] = '64'
#os.environ['OMP_NUM_THREADS'] = '64'

import numpy as np
import numpy.linalg as la
from func_W_hexagon_without_delta_mess_second import readfile
from func_W_hexagon_without_delta_mess_second import H
from func_W_hexagon_without_delta_mess_second import gradF
from func_W_hexagon_without_delta_mess_second import draw
from func_W_hexagon_without_delta_mess_second import innp
import time
start = time.time()
# parameter
#dt = 0.0002
dt = 0.1
#maxiter = 2
maxiter = 50
# minl = 1e-6
# l = 1e-6
minl = 1e-2
l = 1e-2

# read xx from file
filename = "./image/Tq"
#data = readfile(filename)
#xx = data['xx']
# initialise xx
#xx = np.array([0.0,np.cos(np.pi/6)*0.66,np.cos(np.pi/2)*0.66,np.cos(np.pi/6*5)*0.66,0.0,np.sin(np.pi/6)*0.66,np.sin(np.pi/2)*0.66,np.sin(np.pi/6*5)*0.66])
xx = np.array([0.0,np.cos(np.pi/6)*0.66,np.cos(np.pi/2)*0.66,np.cos(np.pi/6*5)*0.66,np.cos(np.pi/6*7)*0.66,0.0,np.sin(np.pi/6)*0.66,np.sin(np.pi/2)*0.66,np.sin(np.pi/6*5)*0.66,np.sin(np.pi/6*7)*0.66])
#xx = np.array([ 0.015,  0.4,  0.4, 0, 0.7, -0.7])
#xx = np.array([0.0, 0.0,  0.56, -0.56])
#xx = np.array([0.0,np.cos(np.pi/6)*0.66,np.cos(np.pi/2)*0.66,0.0, np.sin(np.pi/6)*0.66,np.sin(np.pi/2)*0.66])
#xx = np.array([0.0,np.cos(np.pi/6)*0.66,np.cos(np.pi/2)*0.66,np.cos(np.pi/6*5)*0.66,np.cos(np.pi/6*7)*0.66,np.cos(np.pi/6*9)*0.66, np.cos(np.pi/6*11)*0.66, 0.0,np.sin(np.pi/6)*0.66,np.sin(np.pi/2)*0.66,np.sin(np.pi/6*5)*0.66,np.sin(np.pi/6*7)*0.66,np.sin(np.pi/6*9)*0.66, np.sin(np.pi/6*11)*0.66])
#xx = np.array([0.0,np.cos(np.pi/6)*0.66,np.cos(np.pi/2)*0.66,np.cos(np.pi/6*5)*0.66,np.cos(np.pi/6*7)*0.66,np.cos(np.pi/6*9)*0.66, np.cos(np.pi/6*11)*0.66,0.9,np.cos(np.pi/3)*0.9, np.cos(np.pi*2/3)*0.9, -0.9, np.cos(np.pi*4/3)*0.9, np.cos(np.pi*5/3)*0.9,0.0,np.sin(np.pi/6)*0.66,np.sin(np.pi/2)*0.66,np.sin(np.pi/6*5)*0.66,np.sin(np.pi/6*7)*0.66,np.sin(np.pi/6*9)*0.66, np.sin(np.pi/6*11)*0.66,0.0,np.sin(np.pi/3)*0.9, np.sin(np.pi*2/3)*0.9, 0.0, np.sin(np.pi*4/3)*0.9, np.sin(np.pi*5/3)*0.9])
#xx = np.array([0.0, 0.5, -0.25, -0.25, 0.0, 0.0, np.sqrt(3) / 4, - np.sqrt(3) / 4])
#xx = np.array([0,0.5,0,0])
#xx = np.array([0,0.25,0.25,0,np.sqrt(3)/4,-np.sqrt(3)/4])
#xx = np.array([0,0,0,np.sqrt(3)/4])
n = len(xx)
#data = readfile(filename)
k = 4
# read V from file
#V = data['V'][:, 0:k]
# initialise random V
V = np.random.randn(n, k)
assert xx.shape == (n,), "wrong shape xx"
assert V.shape == (n, k), "wrong shape V"
print("reading complete")
print(xx)
print(V)

#draw(
    #filename=filename,
    #xx=xx,
    #V=V,
#)
for i in range(k):
    V[:, i] = V[:, i] - V[:, :i].dot((V[:, :i].T).dot(V[:, i]))
    V[:, i] = V[:, i] / np.linalg.norm(V[:, i])
alpha = np.zeros(k)
U = np.zeros((n, k))
W = np.zeros((n, k))
Y = np.zeros((n, k))
Vp = np.zeros((n, k))
Up = np.zeros((n, k))
iter = 1
while iter <= maxiter:
    emp = []
    for i in range(k):
        U[:, i] = H(xx, V[:, i], l)
        alpha[i] = V[:, i].dot(U[:,i])
        W[:, i] = U[:, i] - alpha[i] * V[:, i]
        W[:, i] = W[:, i] - V.dot(V.T.dot(W[:,i]))
        W[:, i] = W[:, i] - W[:,:i].dot(W[:,:i].T.dot(W[:,i]))
        nrmW = np.linalg.norm(W[:, i])
        if nrmW > 1e-6:
            W[:, i] = W[:, i] / nrmW
            Y[:, i] = H(xx, W[:, i], l)
        else:
            W[:, i] = np.zeros(n)
            Y[:, i] = np.zeros(n)
            emp.append(k + i)
    print("alpha=v dot u",alpha)

    for i in range(k):
        Vp[:, i] -= np.concatenate((V, W, Vp[:, :i]), axis=1).dot(np.concatenate((V, W, Vp[:, :i]), axis=1).T.dot(Vp[:, i]))
        nrmVpi = np.linalg.norm(Vp[:, i])
        if nrmVpi > 1e-6:
            Vp[:, i] = Vp[:, i] / nrmVpi
            Vp[:, i] = Vp[:, i] - np.concatenate((V, W, Vp[:, :i]), axis=1).dot(np.concatenate((V, W, Vp[:, :i]), axis=1).T.dot(Vp[:, i]))
            Vp[:, i] = Vp[:, i] / np.linalg.norm(Vp[:, i])
            Up[:, i] = H(xx, Vp[:, i], l)
        else:
            Vp[:, i] = np.zeros(n)
            Up[:, i] = np.zeros(n)
            emp.append(2 * k + i)
    UU = np.concatenate((V, W, Vp), axis=1)
    UU = np.delete(UU, emp, axis=1)
    YY = np.concatenate((U, Y, Up), axis=1)
    YY = np.delete(YY, emp, axis=1)
    Pn = UU.T.dot(YY)
    Pn = (Pn + Pn.T) / 2.0
    alpha, eta = la.eig(Pn)
    idx = alpha.argsort()[::1]
    alpha = alpha[idx]
    eta = eta[:, idx]
    Vp = np.copy(V)
    Up = np.copy(U)
    V = np.dot(UU, eta[:, :k])
    alpha = alpha[:k]
    print("alpha=",alpha)
    #if iter % 5 == 0:
    #    print("iter=",iter)
        #draw(
            #filename=filename,
            #xx=xx,
            #V=V,
        #)
    l = max(l / (1 + dt), minl)
    iter = iter + 1
    end = time.time()
    print("time = ", end-start)
print(xx)
print(V)
draw(
    filename=filename,
    xx=xx,
    V=V,
)
