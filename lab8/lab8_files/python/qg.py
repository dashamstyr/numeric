""" Ported by Vlad Popa from Matlab code for EOSC 511, Laboratory 8 files,
    March 2011

    To run:
    >> ipyton -pylab
    >> run qg
    - plots are stored in the plotfiles directory
    - Quicktime can be used to make a movie from png files by using
      File > Open Image Sequence...

    Initial release known issues and to do:
    - saving png plots instead, this could be optimized by writing
      streamfunction array to disk and plotting later
    - could convert to object oriented to allocate memory for
      fields once for the entire run

  output with numpy.io.save or hdf
     http://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html
      http://code.google.com/p/h5py/
      http://www.pytables.org/moin   
"""
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import copy
# inline C code
import scipy.weave
# benchmarking
from time import time
# convolve for Laplacian
from scipy import ndimage
import scipy.ndimage.filters


def param():
    # set the physical parameters of the system
    Av = 1.28e-2           # vertical eddy viscosity (m^2/s)
    H = 500.               # depth (m)
    rho = 1.0e3            # density of water (kg/m^3)
    latitude = 45.         # for calculating parameters of the beta-plane (deg)
    tau_max = 0.2          # wind stress maximum (kg m/s^2)
    b = 2.0e6              # width of the ocean (m)
    a = 2.0e6              # N-S extent of the ocean (m)
    totaltime = 100*86400  # integration time (s)

    # necessary constants
    omega =  7.272205e-05  # rotation rate of the Earth
    earth_radius = 6.4e+6  # radius of the Earth

    # calculate derived parameters
    theta = latitude/360*2*np.pi               #convert degrees to radians
    beta = 2*omega*np.cos(theta)/earth_radius  # beta=df/dy
    f0 = 2*omega*np.sin(theta)                 # Coriolis parameter
    kappa = np.sqrt(Av*f0/2)/H                 # Ekman number = delta/H
    boundary_layer_width_approx = kappa/beta   # estimate of BL width (m)

    # calculation the three nondimensional coefficients
    U0 = tau_max/(b*beta*rho*H)
    epsilon = U0/(b*b*beta)
    wind = 1.
    vis = kappa/(beta*b)
    time = 1./(beta*b)

    # display physycal parameters
    print "Physiscal Paramters:"
    print "a = ", a
    print "b = ", b
    print "totaltime = ", totaltime
    print "epsilon = ", epsilon
    print "wind = ", wind
    print "vis = ", vis
    print "time = ", time
    print "boundary_layer_width_approx = ", boundary_layer_width_approx

    return (b, a, totaltime, epsilon, wind, vis, time)


def numer_init():
    # set up domain
    nx = 16            # number of points in x-direction
    dx = 1./(nx-1)
    ny = int(1./dx+1)

    # set time step
    dt = 43.2e3  # time step  

    # set number of double time steps between plot files
    plotcount = 4 

    # set up the parameters for the relaxation scheme
    tol = 2e-2   # error tolerance
    max = 20     # maximum number of interation
    coeff = 1.7  # relaxation coefficient

    # display simulation parameters
    print "\nSimulation Paramters:"
    print "nx = ", nx
    print "dx = ", dx
    print "ny = ", ny
    print "dt = ", dt
    print "plotcount = ", plotcount
    print "max = ", max
    print "coeff = ", coeff

    return (nx, dx, ny, dt, plotcount, tol, max, coeff)


def vis(psi,nx,ny):
    visc = np.zeros((nx,ny));
    
    # loop version
    # comment out if using 2-D convolution with stencil
    ## for i in range(1, nx-1):
    ##     for j in range(1, ny-1):
    ##         visc[i,j] = psi[i+1,j]+psi[i-1,j]+psi[i,j+1]+psi[i,j-1]-4*psi[i,j]

    # version using 2-D convolution with stencil
    # comment out if using loop
    
    # 5-point centered difference Laplacian stencil
    stencil = np.array([[0., 1., 0.],[1., -4., 1.], [0., 1., 0.]])
    visc = ndimage.convolve(psi, stencil)

    return visc


def mybeta(psi,nx,ny):
    beta = np.zeros((nx,ny))

    # loop version
    # comment out if using 2-D convolution with stencil
    ## for i in range(1,nx-1):
    ##     for j in range(1,ny-1):
    ##         beta[i,j] = psi[i+1,j]-psi[i-1,j]

    # version using 2-D convolution with stencil
    # comment out if using loop
    
    # stencil for beta term
    stencil = np.array([[0., -1., 0.],[0., 0., 0.],[0., 1., 0.]])
    beta = ndimage.convolve(psi, stencil)

    return beta


def jac(psi,vis,nx,ny):
    jaco = np.zeros((nx,ny))

    # loop based python version of code
    # comment out if using inline C code
    ## for i in range(1,nx-1):
    ##     for j in range(1,ny-1):
    ##         # Arakawa Jacobian
    ##         jaco[i,j] =((psi[i+1,j]-psi[i-1,j])*(vis[i,j+1]-vis[i,j-1])- \
    ##                     (psi[i,j+1]-psi[i,j-1])*(vis[i+1,j]-vis[i-1,j])+ \
    ##                     psi[i+1,j]*(vis[i+1,j+1]-vis[i+1,j-1])-psi[i-1,j]* \
    ##                     (vis[i-1,j+1]-vis[i-1,j-1])-psi[i,j+1]* \
    ##                     (vis[i+1,j+1]-vis[i-1,j+1])+psi[i,j-1]* \
    ##                     (vis[i+1,j-1]-vis[i-1,j-1])+vis[i,j+1]* \
    ##                     (psi[i+1,j+1]-psi[i-1,j+1])-vis[i,j-1]* \
    ##                     (psi[i+1,j-1]-psi[i-1,j-1])-vis[i+1,j]* \
    ##                     (psi[i+1,j+1]-psi[i+1,j-1])+vis[i-1,j]* \
    ##                     (psi[i-1,j+1]-psi[i-1,j-1]))*0.33333333
    
    # inline C based version of code
    # comment out if using python code
    code = \
         """
        int i, j;
        for(i = 1; i < nx-1; i++)
           for(j = 1; j < ny-1; j++)
              jaco[i*ny+j] =((psi[(i+1)*ny+j]-psi[(i-1)*ny+j])*(vis[i*ny+j+1]-vis[i*ny+j-1])-
              (psi[i*ny+j+1]-psi[i*ny+j-1])*(vis[(i+1)*ny+j]-vis[(i-1)*ny+j])+psi[(i+1)*ny+j]*
              (vis[(i+1)*ny+j+1]-vis[(i+1)*ny+j-1])-psi[(i-1)*ny+j]*(vis[(i-1)*ny+j+1]-vis[(i-1)*ny+j-1])-
              psi[i*ny+j+1]*(vis[(i+1)*ny+j+1]-vis[(i-1)*ny+j+1])+psi[i*ny+j-1]*(vis[(i+1)*ny+j-1]-
              vis[(i-1)*ny+j-1])+vis[i*ny+j+1]*(psi[(i+1)*ny+j+1]-psi[(i-1)*ny+j+1])-vis[i*ny+j-1]*
              (psi[(i+1)*ny+j-1]-psi[(i-1)*ny+j-1])-vis[(i+1)*ny+j]*(psi[(i+1)*ny+j+1]-psi[(i+1)*ny+j-1])+
              vis[(i-1)*ny+j]*(psi[(i-1)*ny+j+1]-psi[(i-1)*ny+j-1]))*0.33333333;
        """
    scipy.weave.inline(code, ['jaco','psi','vis','nx','ny'])
    
    return jaco


def chi(psi,vis_curr,vis_prev,chi_prev,nx,ny,dx,r_coeff,tol,max_count,epsilon, \
        wind_par,vis_par):
    # calculate right hand side

    beta_term = mybeta(psi,nx,ny)
    wind_term = wind(psi,nx,ny)
    jac_term = jac(psi,vis_curr,nx,ny)
    d = 1./dx

    rhs = -0.5*d*beta_term-epsilon*0.25*d*d*d*d*jac_term+wind_par*0.5*d* \
    wind_term-vis_par*d*d*vis_prev
    (chii,count) = relax(rhs,chi_prev,dx,nx,ny,r_coeff,tol,max_count)
    return (chii,count)


def wind(psi,nx,ny):
    windy = np.zeros((nx,ny))
    tau = np.zeros((nx,ny))

    for i in range(0,nx):
        for j in range(0,ny):
            # fit one negative cosine curve from southern boundary to northern
            tau[i,j] = -np.cos(np.pi*(j-0.5)/(ny-2))

    for i in range(1,nx-1):
        for j in range(1,ny-1):
            windy[i,j] = -tau[i,j+1]+tau[i,j-1]
    
    return windy


def relax(rhs,chi_prev,dx,nx,ny,r_coeff,tol,max_count):
    chi = copy.copy(chi_prev)
    r = np.zeros((nx,ny))

    rr = 1e50
    count = 0

    # loop based python version of code
    # comment out if using inline C code
    ## while (rr>tol) & (count<max_count): 
    ##     chi_max=0.
    ##     r_max = 0.

    ##     for i in range(1,nx-1):
    ##         for j in range(1,ny-1):
    ##             r[i,j] = rhs[i,j]*dx*dx*0.25-((chi[i+1,j]+chi[i,j+1]+chi[i-1,j]+ \
    ##                                            chi[i,j-1])*0.25-chi[i,j])
    ##             if (np.abs(chi[i,j]) > chi_max):
    ##                 chi_max = np.abs(chi[i,j])
    ##             if (np.abs(r[i,j]) > r_max):
    ##                 r_max = np.abs(r[i,j])
    ##             chi[i,j] = chi[i,j]-r_coeff*r[i,j]

    ##     # enforce boundary conditions (zero psi/zero chi) nothing to do.

    ##     if (chi_max==0): 
    ##         rr=1e50 
    ##     else:
    ##         rr=r_max/chi_max
    ##         print "rr= ", rr
    ##     count = count + 1
    
    ## print "count= ", count

    # inline C based version of code
    # comment out if using python code
    # use inline C code
    code = \
         """
         while((rr>tol) && (count<max_count)) {
           float chi_max = 0.0;
           float r_max = 0.0;
           int i, j;
           for(i = 1; i < nx-1; i++)
             for(j = 1; j < ny-1; j++) {
               r[i*ny+j] = rhs[i*ny+j]*dx*dx*0.25-((chi[(i+1)*ny+j]+
                 chi[i*ny+j+1]+chi[(i-1)*ny+j]+chi[i*ny+j-1])*0.25-
                 chi[i*ny+j]);
               if (fabs(chi[i*ny+j]) > chi_max)
                 chi_max = fabs(chi[i*ny+j]);
               if (fabs(r[i*ny+j]) > r_max)
                 r_max = fabs(r[i*ny+j]);
               chi[i*ny+j] = chi[i*ny+j]-r_coeff*r[i*ny+j];
             }

           if (chi_max == 0)
             rr = 1.0e50;
           else
             rr = r_max/chi_max;
             printf("rr= %f \\n", rr);
           count++;
         }
         #"""
    scipy.weave.inline(code, ['nx','ny','dx','tol','max_count', 'r_coeff', 'r', \
                              'rhs', 'chi', 'rr', 'count'])

    return  (chi,count)


if __name__=="__main__":
    # start benchmarking
    start=time()
    
    # initialize the physical parameters
    (pb, pa, ptotaltime, pepsilon, pwind, pvis, ptime) = param()
    # initialize the numerical parameters
    (nnx, ndx, nny, ndt, nplotcount, ntol, nmax, ncoeff) = numer_init()

    # initialize the arrays (need 2 because chi depends on psi at 2 time steps)
    psi_1=np.zeros((nnx,nny))
    psi_2=np.zeros((nnx,nny))

    vis_prev=np.zeros((nnx,nny))
    vis_curr=np.zeros((nnx,nny))

    chii=np.zeros((nnx,nny))
    chi_prev=np.zeros((nnx,nny))

    # non-dimensionalize time
    totaltime = ptotaltime/ptime
    dt = ndt/ptime

    # start time loop
    t = 0
    count = 0
    count_total=0
    plotnum=1

    # set up plotting
    plotdir='plotfiles'
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)
    else:
        #if directory exists, clear the pngfiles
        filenames="%s%s*.png" % (plotdir,os.sep)
        plotfiles=glob.glob(filenames)
        for the_file in plotfiles:
            os.remove(the_file)

    # loop
    while (t < totaltime):
        # write some stuff on the screen so you know code is working
        print '-----------------\n'
        t = t + dt
        print "t= ", t
        
        # update viscosity
        vis_prev = copy.copy(vis_curr)
        vis_curr = vis(psi_1,nnx,nny)
        # find chi, take a step
        (chii,c) = chi(psi_1,vis_curr,vis_prev,chi_prev,nnx,nny,ndx,ncoeff,ntol, \
                       nmax,pepsilon,pwind,pvis)
        psi_2 = psi_2 + dt*chii
        chi_prev = copy.copy(chii)
        count_total=count_total+c
        
        # do exactly the same thing again with opposite psi arrays
        print '-----------------\n'
        t = t + dt
        print "t= ", t
        
        vis_prev = copy.copy(vis_curr)
        vis_curr = vis(psi_2,nnx,nny)
        [chii,c] = chi(psi_2,vis_curr,vis_prev,chi_prev,nnx,nny,ndx,ncoeff,ntol, \
                       nmax,pepsilon,pwind,pvis)
        psi_1 = psi_1 + dt*chii
        chi_prev = copy.copy(chii)

        count_total=count_total+c
        count = count + 1

        # write out psi to a file so it can later be plotted       
        if (nplotcount==count):
            plt.figure()
            plotname = '%s%sPsi%d.png' %(plotdir,os.sep,plotnum)
            plotnum +=1

            # write out to screen the name of the plot file
            print 'Wrote File %s\n' %(plotname)

            plt.contour(np.transpose(psi_1))
            plottitle = 'Results at t = %.0f days' %(t*ptime/86400)
            plt.title(plottitle)
            plt.draw()
            plt.savefig(plotname,dpi=100)
            count = 0
            
    print "count_total", count_total

    # end benchmarking
    print '\nExecution time: ', time()-start, 'seconds'
