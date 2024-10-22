import numpy as np
import time
from func_W_square_without_delta_mess_second import readfile
from func_W_square_without_delta_mess_second import H
from func_W_square_without_delta_mess_second import gradF
from func_W_square_without_delta_mess_second import draw
from func_W_square_without_delta_mess_second import innp

dt=0.01
#dt = 1e-3
#l=1e-3
#minl=1e-3
l = 1e-2
minl = 1e-2
# epsdx=1e-12
epsdx = 1e-6
# epsf=1e-6
epsf = 1e-7
# betat=5
# betau=0.2
betat = 5.0
betau = 0.1
tau=0.01
#tau = 1
gammamax=0.1
gammamin=0.001
#gammamax = 1.0
#gammamin = 0.001
# maxiter = 3e5
maxiter = 100

# read xx from file
filename = "./image/J-I"
data = readfile(filename)
xx = data['xx']

# read V from file
k = 3 # index
V0 = data['V'][:, :k]
print('V0=', V0)
print('xx=', xx)
n = len(xx)
# initialise random V
# V0 = np.random.randn(n, k)
assert xx.shape == (n,), "wrong shape xx"
assert V0.shape == (n, k), "wrong shape V"
print("reading complete")

# calculate eigenvalues alpha, eigenvector V, and index-k state x
V = np.zeros((n,k))
for i in range(k):
    # V[:, i:i + 1] = V0[:, i:i + 1] - np.dot(V[:, 0:i], innp(V[:, 0:i], V0[:, i:i + 1]))
    V[:, i] = V0[:, i] - V[:, :i].dot((V[:, :i].T).dot(V0[:, i]))
    V[:, i] = V[:, i] / np.linalg.norm(V[:, i])
alpha = np.zeros(k)
beta = dt
gamma = np.ones(k) * dt
dV = np.zeros((n, k))
f = gradF(xx)
g = f - 2 * V.dot(V.T.dot(f))
gp = np.copy(g)
Dx = g * beta
xx = xx + Dx
for i in range(k):
    ui = H(xx, V[:, i], l)
    alpha[i] = V[:, i].dot(ui)
    dV[:, i] = -ui + alpha[i] * V[:, i] + V[:, :i].dot(2 * (V[:, :i].T).dot(ui))
print("alpha=", alpha)
dVp = np.copy(dV)
Vp = np.copy(V)
V = V + dV * gamma
for i in range(k):
    V[:, i] -= V[:, :i].dot((V[:, :i].T).dot(V[:, i]))
    V[:, i] /= np.linalg.norm(V[:, i])
DV = V - Vp
l = max(l / (1 + dt), minl)
f = gradF(xx)
norf = np.linalg.norm(f)
print(1, ' ', norf, ' ', alpha, ' ', xx)

iter = 2
start = time.time()
while iter <= maxiter:
    g = f - 2 * V.dot(V.T.dot(f))
    Dg = g - gp
    gp = np.copy(g)
    beta = abs(Dx.dot(Dg) / Dg.dot(Dg))
    beta = min(beta, betat * dt)
    beta = max(beta, betau * dt)
    if norf * beta > tau:
        beta = tau / norf
    # beta = beta * 1e1
    Dx = g * beta
    xx = xx + Dx
    for i in range(k):
        ui = H(xx, V[:, i], l)
        alpha[i] = V[:, i].dot(ui)
        dV[:, i] = -ui + alpha[i] * V[:, i] + V[:, :i].dot(2 * (V[:, :i].T).dot(ui))
    Dd = dVp - dV
    dVp = np.copy(dV)

    gamma = np.abs(np.sum(DV * Dd, axis=0) / np.sum(Dd * Dd, axis=0))
    gamma[~np.isfinite(gamma)] = 1

    np.clip(gamma, gammamin*dt, gammamax*dt)

    Vp = np.copy(V)
    V = V + dV * gamma
    for i in range(k):
        V[:, i] = V[:, i] - V[:, :i].dot((V[:, :i].T).dot(V[:, i]))
        V[:, i] = V[:, i] / np.linalg.norm(V[:, i])
    DV = V - Vp
    l = max(l / (1 + dt), minl)

    f = gradF(xx)
    norf = np.linalg.norm(f)
    if norf < epsf:
        break
    print(iter, ' ', norf, ' ', alpha, ' ', xx)
    iter = iter + 1
    end = time.time()
    print(end - start)

# save results (x,V) in .npz and nematic director plots in .vtu
filename = "./image/J-I_critical_index_3"
draw(
    filename=filename,
    xx=xx,
    V=V,
)

