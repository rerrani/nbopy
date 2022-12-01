#!/usr/bin/env python3
###########################################################################################
##### Code snipped to build Gadget2 compatible HDF5 files for 'nbopy' N-body models #######
##### See https://github.com/rerrani/nbopy for code and documentation               #######
###########################################################################################

from __future__ import print_function
import numpy as np
import h5py

print ("#  USING GADGET G=1 UNITS ")
G = 1.
mU= 1.0e10   # Msol
rU= 1.0      # kpc
tU= 4.714e-3 # Gyrs
vU= 207.4    # km/s

### Loading 'nbopy' N-body model generated using 'nbody.py'
modelname = "model.npy"
data = np.load(modelname)

### Desired total mass and scale radius of the HDF5 model   (here: 1e5 Msol and 100 pc)
M      = 1e5 / mU # Msol - total mass   of the HDF5 model   (the nbopy .npy model has M=1)
rscale = 0.1 / rU # kpc  - scale radius of the HDF5 model   (the nbopy .npy model has rscale=1)

### Desired center coordinates and systemic velocity of the HDF5 model
sysX,  sysY,  sysZ  = np.array(  (  0,  0,  0  )  ) / rU # kpc
sysVX, sysVY, sysVZ = np.array(  (  0,  0,  0  )  ) / vU # km/s

### Re-scaling of the 'nbopy' N-body model (M=1, rscale=1) to the desired mass and size
x =  np.array(data[:,0], dtype="float32") * rscale + sysX
y =  np.array(data[:,1], dtype="float32") * rscale + sysY
z =  np.array(data[:,2], dtype="float32") * rscale + sysZ
vx = np.array(data[:,3], dtype="float32") * np.sqrt(G*M/rscale) + sysVX
vy = np.array(data[:,4], dtype="float32") * np.sqrt(G*M/rscale) + sysVY
vz = np.array(data[:,5], dtype="float32") * np.sqrt(G*M/rscale) + sysVZ

### Some diagnostic output
N = len(x)
partmass = M / float(N)
print ("  (*) Loaded model \'%s\' with N=%i (lg N=%.2f) particles"%(modelname, N, np.log10(N)) )
print ("  (*) Scaling to lg (M/Msol)=%.2f, rs=%.2fpc"%( np.log10(M*mU), 1000* rscale*rU ) )
print ("  (*) Systemic XYZ (kpc)  %.2f, %.2f, %.2f  "%( sysX*rU, sysY*rU, sysZ*rU ) )
print ("  (*) Systemic UVW (km/s) %.2f, %.2f, %.2f  "%( sysVX*vU, sysVY*vU, sysVZ*vU ) )

### Generating the HDF5 file. We first write the Gadget2 compatible header, then the re-scaled coordinates and velocities.
filename = modelname[:-4] + ".hdf5"
f = h5py.File(filename, 'w')

print ("  (*) Writing output hdf5 to \'%s\'"%filename)
print ("  (*) Writing Gadget2 header" )

Header = f.create_group("Header")
Header.attrs.create('BoxSize',                0.,             dtype=np.float64)
Header.attrs.create('Flag_Cooling',           0 ,             dtype=np.int32)
Header.attrs.create('Flag_Entropy_ICs',       (0,0,0,0,0,0),  dtype=np.uint32)
Header.attrs.create('Flag_Feedback',          0 ,             dtype=np.int32)
Header.attrs.create('Flag_Metals',            0 ,             dtype=np.int32)
Header.attrs.create('Flag_Sfr',               0 ,             dtype=np.int32)
Header.attrs.create('Flag_Flag_StellarAge',   0 ,             dtype=np.int32)
Header.attrs.create('HubbleParam',            1.,             dtype=np.float64)
Header.attrs.create('MassTable',              (0.,partmass,0.,0.,0.,0.,), dtype=np.float64)
Header.attrs.create('NumFilesPerSnapshot',    1 ,             dtype=np.int32)
Header.attrs.create('NumPart_ThisFile',       (0,N,0,0,0,0) , dtype=np.int32)
Header.attrs.create('NumPart_Total',          (0,N,0,0,0,0) , dtype=np.uint32)
Header.attrs.create('NumPart_Total_HighWord', (0,0,0,0,0,0) , dtype=np.uint32)
Header.attrs.create('Omega0',                 0.,             dtype=np.float64)
Header.attrs.create('OmegaLambda',            0.,             dtype=np.float64)
Header.attrs.create('Redshift',               0.,             dtype=np.float64)
Header.attrs.create('Time',                   0.,             dtype=np.float64)

PartType1 = f.create_group("PartType1")
Acceleration = PartType1.create_dataset("Acceleration", (N,3), dtype=np.float32)
Coordinates  = PartType1.create_dataset("Coordinates",  (N,3), dtype=np.float32)
ParticleIDs  = PartType1.create_dataset("ParticleIDs",  (N,),  dtype=np.uint32)
Potential    = PartType1.create_dataset("Potential",    (N,),  dtype=np.float32)
Velocities   = PartType1.create_dataset("Velocities",   (N,3), dtype=np.float32)

print ("  (*) Writing coordinates, velocities and IDs" )

# !! write
f[u'PartType1'][ u'Coordinates'][...] = np.column_stack((x,y,z))
# !! write
f[u'PartType1'][ u'Velocities'][...]  = np.column_stack((vx,vy,vz))

f[u'PartType1'][ u'ParticleIDs'][...] = np.arange(N)

print ("  (*) all done :o)" )

f.close()
