#!/usr/bin/env python2
#
#  nbody.py : generates equilibrium N-body models for a given density distribution  
#             with isotropic velocity dispersion using Eddington inversion.
#             see e.g. Spitzer 1987 'dynamical evolution of globular clusters'
#
#  RE May 2019, raer@roe.ac.uk, www.roe.ac.uk/~raer

from __future__ import print_function
from math import pi
import numpy as np
from scipy.integrate import quad
import sys


# seed of the random number generator 
np.random.seed(667408)

# alpha-beta-gamma density profile
def rho(r) :   return r**(-g) * (1. + r**a )**((g-b)/a)  
a = 1.0
b = 4.0
g = 1.0
M =  4 * pi * quad( lambda r: r*r * rho(r) , 0., np.inf)[0] # non-normalized total mass

modelname = "model.npy"
N        = 1e7    # number of particles to draw
Ndraw    = 1e6    # number of random numbers drawn at a time 
NE       = 4e4    # number of energy bins
NR       = 4e4    # number of radius bins
Rmax     = 100    # maximum sampled radius
Rmin     = 1e-2   # minimum sampled radius
R        = np.logspace(np.log10(Rmin),np.log10(Rmax),num=NR)

EPSREL = 1e-6     # integration relative precision


# some feedback on the number of particles within and outside of Rmax,Rmin
Nout = int(N * (1 - 4 * pi * quad( lambda r: r*r * rho(r)/M , 0., Rmax)[0] ) )
Nin  = int(N *      4 * pi * quad( lambda r: r*r * rho(r)/M , 0., Rmin)[0] )
print("     (!) out of %i particles there are %i < DRrmin (%.2f per cent)"%(N,Nin, 100.*Nin/float(N) ) )
print("     (!) and %i > Rmax (%.2f per cent)"%(Nout,  100.*Nout/float(N) ) )

# cumulative mass
Mr  = np.vectorize(lambda x: 4*pi *  quad( lambda r: r*r * rho(r)/M , 0., x  )[0]  )

# potential
Phi = np.vectorize(lambda x: -4*pi * (1/x * quad( lambda r: r*r * rho(r)/M , 0., x    )[0]  +     quad( lambda r: r   * rho(r)/M , x, np.inf)[0] )   )

print("      *  Building Psi and Nu arrays" )
psi = - Phi(R)
nu  =   rho(R) / M
Mcum = Mr(R) 


print("      *  Calculating gradients" )
dndp   = np.gradient(nu,   psi)
d2nd2p = np.gradient(dndp, psi)


print("      *  Evaluating DF" )
f = np.vectorize( lambda e: 1./(np.sqrt(8)*pi*pi) * (quad( lambda p:  np.interp(p, psi[::-1], d2nd2p[::-1]) / np.sqrt(e-p) , 0., e,  epsrel=EPSREL)[0]  ) )


maxE = psi[0]           # most-bound energy
minE = maxE/float(NE)   # least-bound energy
E = np.linspace(minE,maxE,num=NE)
DF = f(E)               # here we build the DF


# check whether DF is physical - if not, check for rounding errors in integration, try increasing/decreasing NR, NE
if np.any (DF < 0) : 
  print("      *  Exit. DF < 0, see df.dat" )
  sys.exit(0)
else: print("      *  DF >= 0, all good" )



print("      *  Differential dN(E)/dE" )
# phase space volume element per energy accessible at given energy and fixed radius 
def dPdr(e,r):  return np.sqrt( 2.*(np.interp(r, R, psi) - e )) * r*r     
# likelihood of a particle to have energy E at a fixed radius
def PLikelihood(e,r): return np.interp(e,E,DF) * dPdr(e,r)
# phase space volume accessible at a given energy, i.e. integrated over all admitted radii
dP = np.vectorize( lambda e: quad( lambda r: dPdr(e,r), 0, np.interp(e,psi[::-1],R[::-1]),   epsrel=EPSREL )[0] ) # rmax = np.interp(e,psi,R)
# differential energy distribution / (4pi)^2
dNdE = DF * dP(E)


print("      *  Finding maximum likelihood" )
maxPLikelihood = []
for RR in R:
  allowed = np.where (E <= np.interp(RR,R,psi) )[0]
  ThismaxPLikelihood = 1.1 * np.amax( PLikelihood(E[allowed],RR) )   # 10 per cent tollerance
  maxPLikelihood.append(ThismaxPLikelihood)
maxPLikelihood = np.array(maxPLikelihood)
  

print("      *  Allocating memory for N-body" )
xx = np.zeros(int(N),dtype="float16")
yy = np.zeros(int(N),dtype="float16")
zz = np.zeros(int(N),dtype="float16")
vx = np.zeros(int(N),dtype="float16")
vy = np.zeros(int(N),dtype="float16")
vz = np.zeros(int(N),dtype="float16")

print("      *  Draw particles" )




n=0             # current number of generated 'particles'
Efails = 0      # rejection sampling failures in Energy


# while we still need to generate 'particles'..
while (n < N):
  # inverse transform sampling for R
  randMcum = np.random.rand(int(Ndraw))      # rand. between 0 and 1
  randR = np.interp(randMcum ,Mcum,R)        # inverse gives radius distributed like rho(r)
  
   # rejection sampling for E
  psiR = np.interp(randR ,R,psi)             # potential at radius
  randE = np.random.rand(int(Ndraw)) * psiR  # random E with constraint that  E > 0 but E < Psi 
  rhoE  = PLikelihood(randE,randR)           # likelihood for E at given R
  randY = np.random.rand(int(Ndraw)) * np.interp(randR,R, maxPLikelihood) 
  Missidx = np.where(randY > rhoE)[0]        # sampled energies rejected
  Efails += len(Missidx)
  
  # repeat sampling at fixed R till we got all the energies we need
  while len(Missidx):
    randE[Missidx] = np.random.rand(len(Missidx)) * psiR[Missidx]  
    rhoE[Missidx]  = PLikelihood(randE[Missidx],randR[Missidx])
    randY[Missidx] = np.random.rand(len(Missidx)) * np.interp(randR[Missidx],R, maxPLikelihood)
    Missidx = np.where(randY > rhoE)[0]
    Efails += len(Missidx)
    
  okEidx = np.where(randY <= rhoE)[0]
  
  if len(okEidx) != int(Ndraw):             # this should never happen.
    print("      *  Particles went missing. Exit." )
    sys.exit(0)
  
  # Let's select as many R,E combinations as we're still missing to get N particles in total
  missing = int(N) - int(n)
  if len(okEidx) <= missing: 
    arraxIdx = n + np.arange(0,len(okEidx))
  else: 
    arraxIdx = n + np.arange(0,missing)
    okEidx = okEidx[:missing]

  
  # spherical symmetric model, draw random points on sphere
  Rtheta  = np.arccos (2. * np.random.rand(len(okEidx)) - 1.)
  Rphi    = np.random.rand(len(okEidx)) * 2*np.pi

  # isotropic velocity dispersion, draw random points on sphere
  Vtheta  = np.arccos (2. * np.random.rand(len(okEidx)) - 1.)
  Vphi    = np.random.rand(len(okEidx)) * 2*np.pi
  V =  np.sqrt( 2.*(psiR[okEidx] - randE[okEidx] ))  
  
  # spherical to cartesian coordinates 
  xx[arraxIdx] = randR[okEidx] * np.sin(Rtheta) * np.cos(Rphi)
  yy[arraxIdx] = randR[okEidx] * np.sin(Rtheta) * np.sin(Rphi)
  zz[arraxIdx] = randR[okEidx] * np.cos(Rtheta) 
  
  vx[arraxIdx] = V * np.sin(Vtheta) * np.cos(Vphi)
  vy[arraxIdx] = V * np.sin(Vtheta) * np.sin(Vphi)
  vz[arraxIdx] = V * np.cos(Vtheta)

  n += len(okEidx)
  print ("         %.2f per cent; E rejection ratio %.2f "%(100* n/float(N) ,  100* Efails/float(n)  ) )

print("      *  Writing output file" )

if modelname[-4:] == ".npy":
  np.save(modelname,np.column_stack((xx,yy,zz,vx,vy,vz)) )   # save npy array
else:
  np.savetxt(modelname,np.column_stack((xx,yy,zz,vx,vy,vz)) )  # ASCII output


print("      *  All done :o)" )
