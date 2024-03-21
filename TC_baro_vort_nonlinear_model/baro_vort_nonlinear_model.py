#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jan 14 18:07:42 2021
Authors: James Penn, Geoffrey K. Vallis

edited by: akshaya C. Nikumbh

"""

from __future__ import (division, print_function)
import matplotlib.pyplot as plt
import numpy as np

from numpy import pi, cos, sin
from numpy.fft import fftshift, fftfreq
from numpy.fft.fftpack import rfft2, irfft2
###########################################
###Functions
def ft(phi):
    """Go from physical space to spectral space."""
    return rfft2(phi, axes=(-2, -1))

def ift(psi):
    """Go from spectral space to physical space."""
    return irfft2(psi, axes=(-2,-1))

def courant_number(psix, psiy, dx, dt):
    """Calculate the Courant Number given the velocity field and step size."""
    maxu = np.max(np.abs(psiy))
    maxv = np.max(np.abs(psix))
    maxvel = maxu + maxv
    return maxvel*dt/dx

def grad(phit):
    """Returns the spatial derivatives of a Fourier transformed variable.
    Returns (∂/∂x[F[φ]], ∂/∂y[F[φ]]) i.e. (ik F[φ], il F[φ])"""
    global ik, il
    phixt = ik*phit        # d/dx F[φ] = ik F[φ]
    phiyt = il*phit        # d/dy F[φ] = il F[φ]
    return (phixt, phiyt)

def anti_alias(phit):
    """Set the coefficients of wavenumbers > k_mask to be zero."""
    k_mask = (8./9.)*(nk+1)**2.
    phit[(np.abs(ksq/(dk*dk)) >= k_mask)] = 0.0
    
    
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)    

def high_wn_filter(phit):
    """Applies the high wavenumber filter of smith et al 2002"""
    filter_dec = -np.log(1.+2.*pi/nk)/((nk-kcut)**filter_exp)
    filter_idx = np.abs(ksq/(dk*dk)) >= kcut**2.
    phit[filter_idx] *= np.exp(filter_dec*(np.sqrt(ksq[filter_idx]/(dk*dk))-kcut)**filter_exp)

def forcing_spatial_mask(phi):
   """TODO: Make spatial mask for forcing"""
   phi[idx_sp_mask] *= 0.

_prhs, _pprhs  = 0.0, 0.0  # previous two right hand sides

def adams_bashforth(zt, rhs, dt):
    """Take a single step forward in time using Adams-Bashforth 3."""
    global step, t, _prhs, _pprhs
    if step is 0:
        # forward euler
        dt1 = dt
        dt2 = 0.0
        dt3 = 0.0
    elif step is 1:
        # AB2 at step 2
        dt1 = 1.5*dt
        dt2 = -0.5*dt
        dt3 = 0.0
    else:
        # AB3 from step 3 on
        dt1 = 23./12.*dt
        dt2 = -16./12.*dt
        dt3 = 5./12.*dt

    newzt = zt + dt1*rhs + dt2*_prhs + dt3*_pprhs    #######_prhs, _pprhs - previous two right hand sides
    _pprhs = _prhs
    _prhs  = rhs
    return newzt
#################################

#############################################################
############################################################
###Velocity and vorticity distribution from Chan and Wiiliams 1987 JAS

###########################################################
rm=100 ### (rnormalized)rm=100 km
vm=40  ###m s-1
b=1
#######Using Chan and William 1987 eqn 2.10, 2.11
r=np.arange(0,1280*4,10)   ########## r=0 to r=1270

v_r=vm*(r/rm)*np.exp((1/b)*(1-(r/rm)**b)) 

z_r=(2*(vm/rm))*(1-((1/2)*(r/rm)**b))*np.exp((1/b)*(1-(r/rm)**b)) ####relative_vorticty eqn 2.11

#z_r[0]=np.nan    ##########To avoid infinity value at centre when r=0

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
ax1.plot(r,v_r,color='r')
ax1.set_xlabel('Distance R from centre(km)')
ax1.set_ylabel(r'velocity($m$ $s^{-1}$)')
ax1.set_title('Velocity distribution')
ax2.plot(r,z_r,color='b')
ax1.set_xlabel('Distance R from centre(km)')
ax2.set_title('Vorticity distribution')
ax2.set_xlabel('Distance R from centre(km)')
ax2.set_ylabel(r'vorticity ($s^{-1}$)')
plt.savefig("Initial velocity and vorticity distribution.png",dpi=300)
###################################
#Convert polar to cartesian coordinates
theta=np.arange(0,np.pi/2,np.pi/(2*128*4))  
vxq1=np.zeros((len(r),len(r)))
vyq1=np.zeros((len(r),len(r)))
zq1=np.zeros((len(r),len(r)))
zq1[:]=np.nan

for r1 in range(0,len(r)):
    for t in range(0,len(theta)):
     
        vxq1[np.int(r1*cos(theta[t])),np.int(r1*sin(theta[t]))]=v_r[r1]*cos(theta[t])
        vyq1[np.int(r1*cos(theta[t])),np.int(r1*sin(theta[t]))]=v_r[r1]*sin(theta[t])
        
        if ~(np.isinf(z_r[r1])):
            zq1[np.int(r1*cos(theta[t])),np.int(r1*sin(theta[t]))]=z_r[r1]

########################################
x=np.arange(0,len(theta),1)
y=np.arange(0,len(theta),1)
xx,yy=np.meshgrid(x,y)
vmq1=np.sqrt(vxq1**2+vyq1**2)
#####################################################################
vxq2=np.copy(vxq1)
vxq3=np.copy(vxq1)
vxq4=np.copy(vxq1)

vyq2=np.copy(vyq1)
vyq3=np.copy(vyq1)
vyq4=np.copy(vyq1)

###############################
########Crosscheck_velocity ditribution for 2 qudrant(vmq1)
vmq2=np.sqrt(vxq2**2+vyq2**2)
##############Get velocity and vorticity distribution for all co-ordinates
vxup=np.hstack((np.fliplr(vxq2),vxq1))
vyup=np.hstack((np.fliplr(vyq2),vyq1))

vxdn=np.hstack((np.fliplr(vxq3),vxq4))
vydn=np.hstack((np.fliplr(vyq3),vyq4))


zrup=np.hstack((np.fliplr(zq1),zq1))
zrdn=np.hstack((np.fliplr(zq1),zq1))
zr1=np.vstack((np.flipud(zrup),zrdn))   ############Vorticity for all coordinates
#
vx1=np.vstack((np.flipud(vxup),vxdn))
vy1=np.vstack((np.flipud(vyup),vydn))

####
##Give cyclonic direction to wind:
##Quad1:
vx=np.zeros((len(theta)*2,len(theta)*2))
vy=np.zeros((len(theta)*2,len(theta)*2))
vx=np.copy(vx1)
vy=np.copy(vy1)
# ###Quad2:rleft: lower; v negative    

vx[256:,256:]=-1*vx1[256:,256:]

###Quad2:rleft: top; bothu and v negative
vx[256:,:256]=-1*vx1[256:,:256]    
vy[256:,:256]=-1*vy1[256:,:256] 

 ###Quad2:rleft: lower; v negative    
vy[:256,:256]=-1*vy1[:256,:256]  
 ##Check velocity distribution for all coodrinates  
v_mag=np.sqrt((vx**2+vy**2))
plt.figure()
plt.imshow(v_mag,origin='lower')
plt.title('Magnitude of Velocity magnitude')

zr1=np.copy(zr1[384:640,384:640])
plt.figure()
plt.imshow(zr1,origin='lower')
plt.title('Magnitude of vorticity magnitude')
#######################
###############################################################################
#zr1[np.isnan(zr1)]=0.0   #####make nan values of the vorticity zero
###############################################################################
######The code below this is based mainly on Prof. Vallis: 
##Further details at https://empslocal.ex.ac.uk/people/staff/gv219/codes/barovort.html
###############################################################################
#Constants and variables

### Configuration
nx = 256
ny = 256                       # numerical resolution
Lx = 1.0
Ly = 1.0                        # domain size [m]
ubar = 0.00                     # background zonal velocity  [m/s]
beta = 12.0                      # beta-plane f = f0 + βy     [1/s 1/m]
n_diss = 2.0                    # Small-scale dissipation of the form ∆^2n_diss, such that n_diss = 2 would be a ∆^4 hyperviscosity.
tau = 0.1                       # coefficient of dissipation
                                # smaller = more dissipation

#Poorly determined coefficients for forcing and dissipation
r_rayleigh = (1./50000.)/np.sqrt(10.)
forcing_amp_factor=100.0/np.sqrt(1.)
r_rayleigh = 0.
forcing_amp_factor=0.


t = 0.0
tmax = 1000
step = 0

ALLOW_SPEEDUP = True         # if True, allow the simulation to take a larger
SPEEDUP_AT_C  = 0.4         # timestep when the Courant number drops below
                             # value of parameter SPEEDUP_AT_C
SLOWDN_AT_C = 0.6            # reduce the timestep when Courant number
                             # is bigger than SLOWDN_AT_C
PLOT_EVERY_S = 100
PLOT_EVERY_S2 = 40

### Physical Domain
nl = ny
nk = nx/2 + 1
dx = Lx / nx
dy = Ly / ny
dt = 0.4 * 16.0 / nx          # choose an initial dt. This will change
                              # as the simulation progresses to maintain
                              # numerical stability
dk = 2.0*pi/Lx
dl = 2.0*pi/Ly
y = np.linspace(0, Ly, num=ny)

y_arr = np.flipud(np.tile(y,(nx,1)).transpose())

xx = np.linspace(0, Lx, num=nx)
yy = 1. - np.linspace(0, Ly, num=ny)
# calculate the wavenumbers [1/m]
# The real FT has half the number of wavenumbers in one direction:
# FT_x[real] -> complex : 1/2 as many complex numbers needed as real signal
# FT_y[complex] -> complex : After the first transform has been done the signal
# is complex, therefore the transformed domain in second dimension is same size
# as it is in euclidean space.
# Therefore FT[(nx, ny)] -> (nx/2, ny)
# The 2D Inverse transform returns a real-only domain (nx, ny)
k = dk*np.arange(0, nk, dtype=np.float64)[np.newaxis, :]
l = dl*fftfreq(nl, d=1.0/nl)[:, np.newaxis]

ksq = k**2 + l**2
ksq[ksq == 0] = 1.0             # avoid divide by zero - set ksq = 1 at zero wavenum
rksq = 1.0 / ksq                # reciprocal 1/(k^2 + l^2)

ik = 1j*k                       # wavenumber mul. imaginary unit is useful
il = 1j*l                       # for calculating derivatives

######################################################################
###Calculate stream function
# initialise the transformed ζ
zt = ft(zr1)     ###zrnew if calculated from the dv_dx -du_dy
#anti_alias(zt)
z=ift(zt)
psit = -rksq * zt 

amp = forcing_amp_factor* np.max(np.abs(zr1))        # calc a reasonable forcing amplitude

# use the x-dimension for reference scale values
nu = ((Lx/(np.floor(nx/3)*2.0*pi))**(2*n_diss))/tau
#High wavenumber filter coefficients. Any waves with wavenumber below kcut are not dissipated at all.
filter_exp = 8.
kcut = 30.

# Spectral Filter as per [Arbic and Flierl, 2003]
wvx = np.sqrt((k*dx)**2 + (l*dy)**2)
spectral_filter = np.exp(-23.6*(wvx-0.65*pi)**4)
spectral_filter[wvx <= 0.65*pi] = 1.0
###########################
#plt.figure()
#plt.pcolormesh(xx,yy,psit)
########################################

#amp = forcing_amp_factor* np.max(np.abs(qi))        # calc a reasonable forcing amplitude

# initialise the storage arrays
#time_arr[0]=t

psit = -rksq * zt           # F[ψ] = - F[ζ] / (k^2 + l^2)
psixt, psiyt = grad(psit)
psix = ift(psixt)
psiy = ift(psiyt)
####################################
### Diagnostic arrays
time_arr = np.zeros(1)
tot_energy_arr = np.zeros(1)
# initialise the storage arrays
time_arr[0]=t

psit = -rksq * zt           # F[ψ] = - F[ζ] / (k^2 + l^2)
psixt, psiyt = grad(psit)
psix = ift(psixt)
psiy = ift(psiyt)

urms=np.sqrt(np.mean(psix**2 + psiy**2))
tot_energy=0.5*urms**2.
tot_energy_arr[0]=tot_energy

############Streamfunction######################
plt.rcParams['contour.negative_linestyle']= 'dashed'
plt.figure()
psi=ift(psit)
cs1=plt.pcolormesh(xx,yy,psi*10**2)
cs=plt.contour(xx,yy,psi*10**2,colors='k')
plt.colorbar(cs1)


###############################################
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
clevs=np.arange(-2,2.1,0.5)
cs=ax1.contourf(xx,yy,z,clevs,cmap='seismic',extend='both')

ax1.contour(xx,yy,z,clevs,colors='k')
#plt.colorbar(cs)
clevs1=np.arange(-2,2.1,0.5)
cbar=fig.colorbar(cs,ax=ax1,ticks=clevs1,orientation='horizontal',extend='both')
cbar.ax.set_xlabel(r'Vorticity($s^{-1}$)')
#ax1.set_xlabel('Distance R from centre(km)')
#ax1.set_ylabel(r'velocity($m$ $s^{-1}$)')
ax1.set_title('Initial vorticity at{:.2f}s dt={:.2f}'.format(t, dt))
clevs=np.arange(-2,2.01,0.5)
cs2=ax2.contourf(xx,yy,psi*10**3,clevs,cmap='seismic',extend='both')
ax2.contour(xx,yy,psi*10**3,clevs,colors='k')
#plt.colorbar(cs)
cbar2=fig.colorbar(cs2,ax=ax2,ticks=clevs1,orientation='horizontal',extend='both')
cbar2.ax.set_xlabel(r'Streamfunction($10^{-3}$ $m^{2}$ $s^{-1}$)')
ax2.set_title('Initial streamfunction')
plt.savefig("Initial velocity and Streamfunction.png",dpi=300)


#########################################################
#show_plot()
tplot = t + PLOT_EVERY_S
tplot2 = t + PLOT_EVERY_S2


#filter_testt=np.ones(zt.shape)
#high_wn_filter(filter_testt) #enables the profile of the high_wn_filter to be compared with deln, the equivalent for hyperviscosity.
#filter_testt_shift = np.fft.fftshift(filter_testt, axes=(0,))


while t < tmax:
    # calculate derivatives in spectral space
    psit = -rksq * zt           # F[ψ] = - F[ζ] / (k^2 + l^2)
    psixt, psiyt = grad(psit)
    zxt, zyt = grad(zt)

    # transform back to physical space for pseudospectral part
    z[:] = ift(zt)
    psix = ift(psixt)
    psiy = ift(psiyt)
    zx =   ift(zxt)
    zy =   ift(zyt)

    # Non-linear: calculate the Jacobian in real space
    # and then transform back to spectral space
    jac = psix * zy - psiy * zx + ubar * zx
    jact = ft(jac)
    #  jact=0.0 ###Linear domain

    # calculate the size of timestep that can be taken
    # (assumes a domain where dx and dy are of the same order)
    c = courant_number(psix, psiy, dx, dt)
    if c >= SLOWDN_AT_C:
        print('DEBUG: Courant No > 0.8, reducing timestep')
        dt = 0.9*dt
    elif c < SPEEDUP_AT_C and ALLOW_SPEEDUP:
        dt = 1.1*dt

    rhs = -jact - beta*psixt    #####Note forcing is zero here, and dissipation is zero too!
    zt[:] = adams_bashforth(zt, rhs, dt)
  #  deln = 1.0 / (1.0 + nu*ksq**n_diss*dt)

   # if t > tplot:
    psi=ift(psit)
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
    clevs=np.arange(-2,2.1,0.5)
    cs=ax1.contourf(xx,yy,z,clevs,cmap='seismic',extend='both')
    
    ax1.contour(xx,yy,z,clevs,colors='k')
    #plt.colorbar(cs)
    clevs1=np.arange(-2,2.1,0.5)
    cbar=fig.colorbar(cs,ax=ax1,ticks=clevs1,orientation='horizontal',extend='both')
    cbar.ax.set_xlabel(r'Vorticity($s^{-1}$)')
    #ax1.set_xlabel('Distance R from centre(km)')
    #ax1.set_ylabel(r'velocity($m$ $s^{-1}$)')
    ax1.set_title('Vorticity at{:.2f}s dt={:.2f}'.format(t, dt))
    clevs=np.arange(-2,2.01,0.5)
    cs2=ax2.contourf(xx,yy,psi*10**3,clevs,cmap='seismic',extend='both')
    ax2.contour(xx,yy,psi*10**3,clevs,colors='k')
    #plt.colorbar(cs)
    cbar2=fig.colorbar(cs2,ax=ax2,ticks=clevs1,orientation='horizontal',extend='both')
    cbar2.ax.set_xlabel(r'Streamfunction($10^{-3}$ $m^{2}$ $s^{-1}$)')
    ax2.set_title('streamfunction')
    plt.savefig('Vorticity and streamfunction at time'+np.str(t)+'dt'+np.str(dt)+'.png',dpi=300)
    t = t + dt
    step = step + 1
    plt.close()
###############################################################################
