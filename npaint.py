#!/usr/bin/env python2
#
#  npaint.py : generates stellar probabilities to paint onto N-body models
#              in the spirit of Bullock & Johnston 2005
#              
#  RE May 2019, raer@roe.ac.uk, www.roe.ac.uk/~raer
#  Corresponding paper: http://arxiv.org/abs/1906.01642 

from __future__ import print_function
from math import pi
import numpy as np
from scipy.integrate import quad
import sys

np.random.seed(667408)

# alpha-beta-gamma density profile, sources potential (think: DM)
def rho(r)  :   return r**(-g)  * (1. + r**a  )**((g-b)/a)  

# alpha-beta-gamma density profile, massless tracers (think: stars)
def rhoS(r) :   return r**(-gS) * (rsS**aS + r**aS )**((gS-bS)/aS)  

# N-body model used to paint stars onto, with density as defined by rho(r)
modelname = "model.npy" 

N = 1e7    # Number of particles in model
a = 1.
b = 4.0
g = 1.
M =  4 * pi * quad( lambda r: r*r * rho(r) , 0., np.inf)[0] # non-normalized total mass

# Stellar tracer density profile
aS  = 2.
bS  = 5.0
gS  = 0.1   # NB needs to be >0, see Appendix A of EP19
rsS = 0.2   # ratio of stellar and DM scale radius
MS  =  4 * pi * quad( lambda r: r*r * rhoS(r) , 0., np.inf)[0] # non-normalized total mass


NE       = 1e4    # number of energy bins
NR       = 1e4    # number of radius bins
Rmax     = 100    # maximum sampled radius
Rmin     = 1e-2   # minimum sampled radius
R        = np.logspace(np.log10(Rmin),np.log10(Rmax),num=NR)


EPSREL = 1e-6  # integration relative precision

# some feedback on the number of particles within and outside of Rmax,Rmin
# DM
Nout = int(N * (1. - 4. * pi * quad( lambda r: r*r * rho(r)/M , 0., Rmax)[0] ) )
Nin  = int(N *       4. * pi * quad( lambda r: r*r * rho(r)/M , 0., Rmin)[0] )
print("     (!) out of %i particles there are %i < Rmin (%.2f per cent)"%(N,Nin, 100.*Nin/float(N) ) )
print("     (!) and %i > Rmax (%.2f per cent)"%(Nout,  100.*Nout/float(N) ) )

# STARS
NoutS =   1. - 4. * pi * quad( lambda r: r*r * rhoS(r)/MS , 0., Rmax)[0] 
NinS  =        4. * pi * quad( lambda r: r*r * rhoS(r)/MS , 0., Rmin)[0] 
print("     (!) Stars: there are %.2f per cent < Rmin "%NinS )
print("     (!) and %.2f per cent > Rmax "%NoutS )


# cumulative DM mass
Mr   = np.vectorize(lambda x: 4*pi *  quad( lambda r: r*r * rho(r)/M , 0., x  )[0]  )
# cumulative stellar mass
MSr  = np.vectorize(lambda x: 4*pi *  quad( lambda r: r*r * rhoS(r)/MS , 0., x  )[0]  )
# potential, sourced by DM alone
Phi  = np.vectorize(lambda x: -4*pi * (1/x * quad( lambda r: r*r * rho(r)/M , 0., x    )[0]  +     quad( lambda r: r   * rho(r)/M , x, np.inf)[0] )   )


print("      *  Building Psi and Nu arrays" )
psi  = - Phi(R)
nuS  =   rhoS(R) / MS
nuDM =   rho(R)  / M

Mcum  = Mr(R)   # DM cumulative mass
MScum = MSr(R)  # stellar cumulative mass

print("      *  Calculating gradients" )
dndpDM   = np.gradient(nuDM,   psi)
d2nd2pDM = np.gradient(dndpDM, psi)
dndpS   = np.gradient(nuS,   psi)
d2nd2pS = np.gradient(dndpS, psi)


print("      *  Evaluating DFs (stars + DM)" )
fS = np.vectorize( lambda e: 1./(np.sqrt(8)*pi*pi) * (quad( lambda p:  np.interp(p, psi[::-1], d2nd2pS[::-1]) / np.sqrt(e-p) , 0., e,  epsrel=EPSREL)[0]  ) ) # + np.interp(0., psi, dndp) / np.sqrt(e)   == 0 due to B.C.
fDM = np.vectorize( lambda e: 1./(np.sqrt(8)*pi*pi) * (quad( lambda p:  np.interp(p, psi[::-1], d2nd2pDM[::-1]) / np.sqrt(e-p) , 0., e,  epsrel=EPSREL)[0]  ) ) # + np.interp(0., psi, dndp) / np.sqrt(e)   == 0 due to B.C.


maxE = psi[0]           # most-bound energy
minE = maxE/float(NE)   # least-bound energy
E = np.linspace(minE,maxE,num=NE)


print("      *    ...stars..." )
DFS = fS(E)             # stellar DF
if np.any (DFS < 0) :   # check if physical. if not, check resolution and gamma_min (gS)
  print("      *  Exit. star DF < 0, see df.dat. NO DM DF computed yet." )  
  np.savetxt("df.dat", np.column_stack(( E, DFS)))
  sys.exit(0)  


print("      *    ...DM..." )
DFDM = fDM(E)           # DM DF
if np.any (DFDM < 0) :  # check if physical
  print("      *  Exit. DM DF < 0, see df.dat" )
  np.savetxt("df.dat", np.column_stack(( E, DFS, DFDM)))
  sys.exit(0)

print("      *  star + dm DF >= 0, all good" )


print("      *  Loading model, calculating energies" )
if modelname[-4:] == ".npy": data = np.load(modelname)
else: data = np.loadtxt(modelname) # ASCII
RR = np.sqrt( data[:,0]**2. + data[:,1]**2. + data[:,2]**2. )
VV = np.sqrt( data[:,3]**2. + data[:,4]**2. + data[:,5]**2. )
EE = np.interp(RR, R, psi) - 0.5*VV**2.


print("      *  Computing probabilities" )
probs = np.interp(EE, E, DFS)/ np.interp(EE, E, DFDM)
norm = np.sum(probs)
probs = probs/norm

print("      *  Writing probability file stars.npy -- to be used with specific model.dat" )
if modelname[-4:] == ".npy": 
  np.save("stars.npy", probs )       # save npy array
else:
  np.savetxt("stars.dat", probs )    # ASCII output


print("      *  All done :o)" )
