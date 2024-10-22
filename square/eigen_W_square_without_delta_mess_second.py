import os
#os.environ['OPENBLAS_NUM_THREADS'] = '64'
#os.environ['OMP_NUM_THREADS'] = '64'

import numpy as np
import numpy.linalg as la
from func_W_square_without_delta_mess_second import readfile
from func_W_square_without_delta_mess_second import H
from func_W_square_without_delta_mess_second import gradF
from func_W_square_without_delta_mess_second import draw
from func_W_square_without_delta_mess_second import innp
import time
start = time.time()
# parameter
#dt = 0.0002
dt = 0.1
#maxiter = 0
maxiter = 50
# minl = 1e-6
# l = 1e-6
minl = 1e-2
l = 1e-2

# read xx from file
filename = "./image/J-I"
data = readfile(filename)
xx = data['xx']
# initialise xx
#xx = np.array([-0.35, 0.5, 0.2, 0.35, -0.2, -0.5])
n = len(xx)
k = 4 # number of eigenvectors
# read V from file
V = data['V'][:, 0:k]
# initialise random V
#V = np.random.randn(n, k)
assert xx.shape == (n,), "wrong shape xx"
assert V.shape == (n, k), "wrong shape V"
print("reading complete")
print(xx)
print(V)

# calculate eigenvalue alpha, eigenvector V
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
    alpha = np.real(alpha)
    eta = np.real(eta)
    idx = alpha.argsort()[::1]
    alpha = alpha[idx]
    eta = eta[:, idx]
    Vp = np.copy(V)
    Up = np.copy(U)
    V = np.dot(UU, eta[:, :k])
    alpha = alpha[:k]
    print("alpha=",alpha)
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
