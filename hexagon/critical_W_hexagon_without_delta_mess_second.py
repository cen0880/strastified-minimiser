import numpy as np
import time
from func_W_hexagon_without_delta_mess_second import readfile
from func_W_hexagon_without_delta_mess_second import H
from func_W_hexagon_without_delta_mess_second import gradF
from func_W_hexagon_without_delta_mess_second import draw
from func_W_hexagon_without_delta_mess_second import innp

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
# tau=0.01
tau = 1
#gammamax=0.1
# gammamin=0.001
gammamax = 1.0
gammamin = 0.001
# maxiter = 3e5
maxiter = 100

# read xx from file
filename = "./image/Tt"
#filename = "./image/H_star"
data = readfile(filename)
xx = data['xx']
k = 1
# read V from file
#V0 = data['V'][:, [0,1]]
V0 = data['V'][:, :k]
print('V0=', V0)
#print('pert=',data['V'][:,3])
#xx = np.array(xx + 0.1*data['V'][:,3])
print('xx=', xx)
n = len(xx)
# initialise random V
# V = np.random.randn(n, k)
assert xx.shape == (n,), "wrong shape xx"
assert V0.shape == (n, k), "wrong shape V"
print("reading complete")
#filename = "./image/H_3+_critical_index_3"
#draw(
    #filename=filename,
    #xx=xx,
    #V=V0
#)

#f = gradF(xx)
#norf = np.linalg.norm(f)
#print(norf)
#xx = np.array([0.0,np.cos(np.pi/6),np.cos(np.pi/2),np.cos(np.pi/6*5),np.cos(np.pi/6*7),np.cos(np.pi/6*9), np.cos(np.pi/6*11), 0.0,np.sin(np.pi/6),np.sin(np.pi/2),np.sin(np.pi/6*5),np.sin(np.pi/6*7),np.sin(np.pi/6*9), np.sin(np.pi/6*11)])*0.7
#print(xx)
#f = gradF(xx)
#norf = np.linalg.norm(f)
#print(norf)
#xx = np.array([0.0,np.cos(np.pi/6),np.cos(np.pi/2),np.cos(np.pi/6*5),np.cos(np.pi/6*7),np.cos(np.pi/6*9), np.cos(np.pi/6*11), 0.0,np.sin(np.pi/6),np.sin(np.pi/2),np.sin(np.pi/6*5),np.sin(np.pi/6*7),np.sin(np.pi/6*9), np.sin(np.pi/6*11)])*0.65
#print(xx)
#f = gradF(xx)
#norf = np.linalg.norm(f)
#print(norf)
#xx = np.array([0.0,np.cos(np.pi/6),np.cos(np.pi/2),np.cos(np.pi/6*5),np.cos(np.pi/6*7),np.cos(np.pi/6*9), np.cos(np.pi/6*11), 0.0,np.sin(np.pi/6),np.sin(np.pi/2),np.sin(np.pi/6*5),np.sin(np.pi/6*7),np.sin(np.pi/6*9), np.sin(np.pi/6*11)])*0.6
#print(xx)
#f = gradF(xx)
#norf = np.linalg.norm(f)
#print(norf)
#xx = np.array([0.0,np.cos(np.pi/6),np.cos(np.pi/2),np.cos(np.pi/6*5),np.cos(np.pi/6*7),np.cos(np.pi/6*9), np.cos(np.pi/6*11), 0.0,np.sin(np.pi/6),np.sin(np.pi/2),np.sin(np.pi/6*5),np.sin(np.pi/6*7),np.sin(np.pi/6*9), np.sin(np.pi/6*11)])*0.55
#print(xx)
#f = gradF(xx)
#norf = np.linalg.norm(f)
#print(norf)
#xx = np.array([0.0,np.cos(np.pi/6),np.cos(np.pi/2),np.cos(np.pi/6*5),np.cos(np.pi/6*7),np.cos(np.pi/6*9), np.cos(np.pi/6*11), 0.0,np.sin(np.pi/6),np.sin(np.pi/2),np.sin(np.pi/6*5),np.sin(np.pi/6*7),np.sin(np.pi/6*9), np.sin(np.pi/6*11)])*0.5
#print(xx)
#f = gradF(xx)
#norf = np.linalg.norm(f)
#print(norf)
#xx = np.array([0.0,np.cos(np.pi/6),np.cos(np.pi/2),np.cos(np.pi/6*5),np.cos(np.pi/6*7),np.cos(np.pi/6*9), np.cos(np.pi/6*11), 0.0,np.sin(np.pi/6),np.sin(np.pi/2),np.sin(np.pi/6*5),np.sin(np.pi/6*7),np.sin(np.pi/6*9), np.sin(np.pi/6*11)])*0.45
#print(xx)
#f = gradF(xx)
#norf = np.linalg.norm(f)
#print(norf)
# initialise xx
# xx = np.array([0.55 * np.sqrt(3) / 2, -0.55 * np.sqrt(3) / 2, 0.55 * 0.5, -0.55 * 0.5])
# filename = "./image/angle_T135.h5"
# data = readfile(filename)
# x = data['xx']
# V = data['VV']

#V = np.random.randn(n, k)
V = np.zeros((n,k))

for i in range(k):
    # V[:, i:i + 1] = V0[:, i:i + 1] - np.dot(V[:, 0:i], innp(V[:, 0:i], V0[:, i:i + 1]))
    V[:, i] = V0[:, i] - V[:, :i].dot((V[:, :i].T).dot(V0[:, i]))
    V[:, i] = V[:, i] / np.linalg.norm(V[:, i])
alpha = np.zeros(k)
beta = dt
# gamma = np.ones((1, k)) * dt
gamma = np.ones(k) * dt
dV = np.zeros((n, k))
f = gradF(xx)
# g = f - 2 * np.dot(V, innp(V, f))
g = f - 2 * V.dot(V.T.dot(f))
gp = np.copy(g)
Dx = g * beta
xx = xx + Dx
for i in range(k):
    ui = H(xx, V[:, i], l)
    # alpha[i:i + 1, 0:1] = innp(V[:, i:i + 1], ui)
    # dV[:, i:i + 1] = -ui + alpha[i:i + 1, 0:1] * V[:, i:i + 1] + np.dot(V[:, 0:i], 2 * innp(V[:, 0:i], ui))
    # alpha[i] = innp(V[:, i], ui)
    alpha[i] = V[:, i].dot(ui)
    dV[:, i] = -ui + alpha[i] * V[:, i] + V[:, :i].dot(2 * (V[:, :i].T).dot(ui))
print("alpha=", alpha)
dVp = np.copy(dV)

Vp = np.copy(V)
V = V + dV * gamma

# V, _ = np.linalg.qr(V)
for i in range(k):
    V[:, i] -= V[:, :i].dot((V[:, :i].T).dot(V[:, i]))
    V[:, i] /= np.linalg.norm(V[:, i])

DV = V - Vp
l = max(l / (1 + dt), minl)
f = gradF(xx)
norf = np.linalg.norm(f)
print(1, ' ', norf, ' ', alpha, ' ', xx)
# [0,0]
iter = 2
start = time.time()
while iter <= maxiter:
    # g = f - 2 * np.dot(V, innp(V, f))
    g = f - 2 * V.dot(V.T.dot(f))
    Dg = g - gp
    gp = np.copy(g)
    # beta = (abs(innp(Dx, Dg) / innp(Dg, Dg)))[0, 0]
    beta = abs(Dx.dot(Dg) / Dg.dot(Dg))
    beta = min(beta, betat * dt)
    beta = max(beta, betau * dt)
    if norf * beta > tau:
        beta = tau / norf
    # beta = beta * 1e1
    Dx = g * beta
    xx = xx + Dx
    # print "g", g[0:k,0:1]
    # print "beta3", beta
    for i in range(k):
        ui = H(xx, V[:, i], l)
        # alpha[i:i + 1, 0:1] = innp(V[:, i:i + 1], ui)
        # dV[:, i:i + 1] = -ui + alpha[i:i + 1, 0:1] * V[:, i:i + 1] + np.dot(V[:, 0:i], 2 * innp(V[:, 0:i], ui))
        alpha[i] = V[:, i].dot(ui)
        dV[:, i] = -ui + alpha[i] * V[:, i] + V[:, :i].dot(2 * (V[:, :i].T).dot(ui))
    Dd = dVp - dV
    dVp = np.copy(dV)

    # for i in range(k):
    #     # gamma[0:1, i:i + 1] = abs(innp(DV[:, i:i + 1], Dd[:, i:i + 1]) / innp(Dd[:, i:i + 1], Dd[:, i:i + 1]))
    #     gamma[i] = abs(DV[:, i].dot(Dd[:, i]) / Dd[:, i].dot(Dd[:, i]))

    #     if not np.isfinite(gamma[0:1, i:i + 1]):
    #         gamma[0:1, i:i + 1] = 1
    # # print "gamma", gamma[0:1,0:k]
    # for i in range(k):
    #     if gamma[0:1, i:i + 1] > gammamax * dt:
    #         gamma[0:1, i:i + 1] = gammamax * dt
    #     if gamma[0:1, i:i + 1] < gammamin * dt:
    #         gamma[0:1, i:i + 1] = gammamin * dt
    # if not np.isfinite(gamma[i]):
    #     gamma[i] = 1

    gamma = np.abs(np.sum(DV * Dd, axis=0) / np.sum(Dd * Dd, axis=0))
    gamma[~np.isfinite(gamma)] = 1

    # limit the values in gamma with min and max
    # for i in range(k):
    #     if gamma[i] > gammamax * dt:
    #         gamma[i] = gammamax * dt
    #     if gamma[i] < gammamin * dt:
    #         gamma[i] = gammamin * dt
    np.clip(gamma, gammamin*dt, gammamax*dt)

    # gamma = min (gamma, gammamax*dt)
    # gamma = max (gamma, gammamin*dt)
    Vp = np.copy(V)
    V = V + dV * gamma
    # print "V", V[0:3,0:k]
    # print "dV", dV[0:7,0:k].T
    for i in range(k):
        # V[:, i:i + 1] = V[:, i:i + 1] - np.dot(V[:, 0:i], innp(V[:, 0:i], V[:, i:i + 1]))
        V[:, i] = V[:, i] - V[:, :i].dot((V[:, :i].T).dot(V[:, i]))
        V[:, i] = V[:, i] / np.linalg.norm(V[:, i])
    DV = V - Vp
    l = max(l / (1 + dt), minl)

    f = gradF(xx)
    norf = np.linalg.norm(f)
    # [0,0]
    if norf < epsf:
        break
    #if iter % 50 == 0:
        #filename = "./image/Tq_critical"
        #draw(
            #filename=filename,
            #xx=xx,
            #V=V,
        #)
    print(iter, ' ', norf, ' ', alpha, ' ', xx)
    iter = iter + 1
    end = time.time()
    print(end - start)
filename = "./image/Tt_critical_index_1"
draw(
    filename=filename,
    xx=xx,
    V=V,
)

