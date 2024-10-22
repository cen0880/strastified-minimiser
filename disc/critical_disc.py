import numpy as np
import numpy.linalg as la
import time
from func_critical_disc import readfilex
from func_critical_disc import readfileV
from func_critical_disc import F
from func_critical_disc import H
from func_critical_disc import innp
from func_critical_disc import nor
from func_critical_disc import draw

# dt=0.0002
dt=1.0
# l=1e-6
# minl=1e-6
l=1e-1
minl=1e-1
# epsdx=1e-12
epsdx=1e-6
# epsf=1e-6
epsf=1e-7
# betat=5
# betau=0.2
betat=5.0
betau=0.1
# tau=0.01
tau=1
# gammamax=0.1
# gammamin=0.001
gammamax=1.0
gammamin=0.001
# maxiter = 3e5
maxiter = 1e5

x = readfilex()
n = x.shape[0]
Vresult = readfileV(2)
#find a critical point with saddle-index k
k = 2
V0 = Vresult[:,0:2]
# x = x + 0.001*Vresult[:,5:6] 
#x = x + 1.0*Vresult[:,3:4] 

#V = np.random.randn(n,k)
V = np.zeros((n,k),dtype = float)

for i in np.arange(k):
    V[:,i:i+1] = V0[:,i:i+1] - np.dot(V[:,0:i],innp(V[:,0:i], V0[:,i:i+1]))
    V[:,i:i+1] = V[:,i:i+1]/nor(V[:,i:i+1])
alpha = np.zeros((k,1),dtype = float)
beta = dt
gamma = np.ones((1,k),dtype = float) * dt
dV = np.zeros((n,k),dtype = float)
f = F (x)
g = f - 2 * np.dot(V,innp(V,f))
gp = np.copy(g)
Dx = g * beta
ui = H(x, V[:,0:0+1], l)
x = x + Dx
for i in np.arange(k):
    ui = H(x, V[:,i:i+1], l)
    alpha[i:i+1,0:1] = innp(V[:,i:i+1], ui)
    dV[:,i:i+1]= -ui + alpha[i:i+1,0:1] * V[:,i:i+1] + np.dot(V[:,0:i], 2 * innp(V[:,0:i],ui))
dVp = np.copy(dV)

Vp = np.copy(V)
V = V + dV * gamma
for i in np.arange(k):
    V[:,i:i+1] = V[:,i:i+1] - np.dot(V[:,0:i],innp(V[:,0:i], V[:,i:i+1]))
    V[:,i:i+1] = V[:,i:i+1]/nor(V[:,i:i+1])
DV = V - Vp
l = max( l/(1+dt), minl)
f = F(x)
norf = nor(f)[0,0]
iter = 2
start = time.time()
while iter<=maxiter:
    g = f - 2 * np.dot(V,innp(V,f))
    Dg = g - gp
    gp = np.copy(g)
    beta = (abs ( innp(Dx,Dg) / innp(Dg,Dg) ))[0,0]
    beta = min ( beta, betat * dt)
    beta = max ( beta, betau * dt)
    if norf*beta > tau:
        beta=tau/norf
    # beta = beta * 1e1
    Dx = g * beta
    x = x + Dx
    # print "g", g[0:k,0:1]
    # print "beta3", beta
    for i in np.arange(k):
        ui = H (x, V[:,i:i+1], l)
        alpha[i:i+1,0:1] = innp (V[:,i:i+1], ui);
        dV[:,i:i+1]= -ui + alpha[i:i+1,0:1] * V[:,i:i+1] + np.dot(V[:,0:i], 2 * innp(V[:,0:i],ui))
    Dd = dVp - dV
    dVp = np.copy(dV)
     
    for i in np.arange(k):
        gamma[0:1,i:i+1] = abs ( innp(DV[:,i:i+1],Dd[:,i:i+1]) / innp(Dd[:,i:i+1],Dd[:,i:i+1]) )
        if ~np.isfinite(gamma[0:1,i:i+1]):
            gamma[0:1,i:i+1]=1
    # print "gamma", gamma[0:1,0:k]
    for i in np.arange(k):
        if gamma[0:1,i:i+1]>gammamax*dt:
            gamma[0:1,i:i+1] = gammamax*dt
        if gamma[0:1,i:i+1]<gammamin*dt:
            gamma[0:1,i:i+1] = gammamin*dt
    # print "beta, gamma", beta, gamma[0:1,0:k]
    # gamma = min (gamma, gammamax*dt)
    # gamma = max (gamma, gammamin*dt)
    Vp = np.copy(V)
    V = V + dV * gamma
    # print "V", V[0:3,0:k]
    # print "dV", dV[0:7,0:k].T
    for i in np.arange(k):
        V[:,i:i+1] = V[:,i:i+1] - np.dot(V[:,0:i],innp(V[:,0:i], V[:,i:i+1]))
        V[:,i:i+1] = V[:,i:i+1]/nor(V[:,i:i+1])
    DV = V - Vp
    l=max(l/(1+dt),minl)
    
    f = F (x)
    norf = nor(f)[0,0]
    if norf < epsf:
        break
    if iter%500==0:
        draw(x,0)
    if iter%100==0:
        print (iter, ' ', norf, ' ', alpha.T)
    print(iter, ' ', norf, ' ', alpha.T)
    iter=iter+1;
    
end = time.time()
print (end-start)

draw(x,0)
for i in np.arange(k):
    draw(V[:,i:i+1],i+1)
# output = struct('x', x, 'V', V, 'it', iter, 'al', alpha);
# if options.outputX~=0
    # output.X=X;
# end
