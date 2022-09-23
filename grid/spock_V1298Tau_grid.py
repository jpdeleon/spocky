#!/usr/bin/env python
# coding: utf-8

import rebound
from spock import FeatureClassifier
feature_model = FeatureClassifier()

from scipy.stats import uniform, norm, truncnorm
import astropy.units as u
import numpy as np

Nsamples = 1_000

Mstar = 1.095 # + 0.049 - 0.047
planets = "b c d e".split()
Nplanets = len(planets)
#b,c,d,e
pers = [24.1399, 8.24892, 12.4058, 40.2]
pers_err = [0.0015, 0.00083, 0.0018, 1.0]

eccs = [0.134, 0.30, 0.20, 0.10]
eccs_err = [0.075, np.nan, np.nan, 0.091]

mps = [0.64, 0.24, 0.31, 1.16]
mps_err = [0.19, np.nan, np.nan, 0.30]


import astropy.units as u
Mj2Ms = u.Mjup.to(u.Msun)

def setup_sim(mps, pers, eccs):
    assert len(mps)==len(pers)==len(eccs)==Nplanets
    sim = rebound.Simulation()
    sim.units = ["msun", "m", "d"]
    sim.add(m=Mstar)
    for i in range(Nplanets):
        sim.add(m=mps[i]*Mj2Ms, P=pers[i], e=eccs[i])
    sim.move_to_com()
    return sim


#setup_sim(mps,pers,eccs)


# ## Generate samples
# ### mass

mvec=[]
for i in range(Nplanets):
    mu, sigma = mps[i], mps_err[i]
    if sigma is np.nan:
        v=uniform(loc=0, scale=mu).rvs(size=Nsamples)*Mj2Ms
    else:
        lower = 1e-6
        upper = mu+5*sigma
        a = (lower - mu) / sigma
        b = (upper - mu) / sigma
        v=truncnorm(a=a, b=b, loc=mu, scale=sigma).rvs(size=Nsamples)*Mj2Ms
    mvec.append(v)
mvec=np.c_[mvec].T

### period
pvec=[norm(loc=pers[i], scale=pers_err[i]).rvs(size=Nsamples) for i in range(Nplanets)]
pvec=np.c_[pvec].T

# ### ecc
evec=[]
for i in range(Nplanets):
    mu = eccs[i]
    sigma = eccs_err[i]
    if sigma is np.nan:
        #upper limit
        v=uniform(loc=0, scale=mu).rvs(size=Nsamples)
    else:
        lower = 0
        upper = 1
        a = (lower - mu) / sigma
        b = (upper - mu) / sigma
        v=truncnorm(a=a, b=b, loc=mu, scale=sigma).rvs(size=Nsamples)
    evec.append(v)
evec=np.c_[evec].T

import pandas as pd
import matplotlib.pyplot as pl

fig,axes = pl.subplots(Nplanets, 3, sharex='col', constrained_layout=True)

fac = u.Msun.to(u.Mjup)
for i,(lbl,par) in enumerate(zip(["mass","period","ecc"],[mvec*fac,pvec,evec])):
    df = pd.DataFrame(par, columns=planets)
    for j,p in enumerate(planets):
        ax=axes[j,i]
        df[p].plot.kde(ax=ax)
        if j==0:
            ax.set_title(lbl)
        if i==0:
            ax.set_ylabel(p)       
        else:
            ax.set_ylabel("")       

fig.savefig("sample_space.png", bbox_inches="tight")

def evenly_select(arr, M):
    "select evenly-spaced sample"
    N = len(arr)
    if M > N/2:
        cut = np.zeros(N, dtype=bool)
        q, r = divmod(N, N-M)
        indices = [q*i + min(i, r) for i in range(N-M)]
        cut[indices] = True
    else:
        cut = np.ones(N, dtype=bool)
        q, r = divmod(N, M)
        indices = [q*i + min(i, r) for i in range(M)]
        cut[indices] = False

    return arr[~cut]

Nsamples = 2

mvec2,pvec2,evec2 = [],[],[]
for p in range(Nplanets):
    s = np.sort(mvec[:, p])
    ss=evenly_select(s, Nsamples)
    mvec2.append(ss)
    
    s = np.sort(pvec[:, p])
    ss=evenly_select(s, Nsamples)
    pvec2.append(ss)
    
    s = np.sort(evec[:, p])
    ss=evenly_select(s, Nsamples)
    evec2.append(ss)
    
mvec2 = np.array(mvec2)
pvec2 = np.array(pvec2)
evec2 = np.array(evec2)


m1s = mvec2[0]
m2s = mvec2[1]
m3s = mvec2[2]
m4s = mvec2[3]

p1s = pvec2[0]
p2s = pvec2[1]
p3s = pvec2[2]
p4s = pvec2[3]

e1s = evec2[0]
e2s = evec2[1]
e3s = evec2[2]
e4s = evec2[3]


c=0
sims=[]
for m1 in m1s:
    for m2 in m2s:
        for m3 in m3s:
            for m4 in m4s:
                for p1 in p1s:
                    for p2 in p2s:
                        for p3 in p3s:
                            for p4 in p4s:
                                for e1 in e1s:
                                    for e2 in e2s:
                                        for e3 in e3s:
                                            for e4 in e4s:
                                                sims.append(setup_sim((m1,m2,m3,m4),
                                                                      (p1,p2,p3,p4),
                                                                      (e1,e2,e3,e4))
                                                           )
#                                                 print(c)
#                                                 print(f"m=({m1:.6f},{m2:.6f},{m3:.6f},{m4:.6f})")
#                                                 print(f"p=({p1:.6f},{p2:.6f},{p3:.6f},{p4:.6f})")
#                                                 print(f"e=({e1:.6f},{e2:.6f},{e3:.6f},{e4:.6f})")
                                                c+=1
                                                

probs = feature_model.predict_stable(sims)


