try:
    from lab5_funs import Integrator
except ImportError:
    import os,sys
    libdir=os.path.abspath('../')
    sys.path.append(libdir)
    from lab5_funs import Integrator

from lab5_funs import Integrator

class Integ52(Integrator):
    """rewrite the init and derivs5 methods to
       work with a single (grey) daisy
    """
    def __init__(self,coeffFileName):
        Integrator.__init__(self,coeffFileName)
        i=self.initVars
        i.yinit=np.array([i.greyConc])
        i.nVars=len(i.yinit)

    def derivs5(self,y,t):
        """y[0]=fraction grey daisies
        """
        sigma=5.67e-8  #Stefan Boltzman constant W/m^2/K^4
        u=self.userVars
        x = 1.0 - y[0]
        albedo_p = x*u.albedo_ground + y[0]*u.albedo_grey
        Te_4 = u.S0/4.0*u.L*(1.0 - albedo_p)/sigma
        eta = u.R*u.S0/(4.0*sigma)
        temp_g = (eta*(albedo_p - u.albedo_grey) + Te_4)**0.25
        if(temp_g >= 277.5  and temp_g <= 312.5): 
            beta_g= 1.0 - 0.003265*(295.0 - temp_g)**2.0
        else:
            beta_g=0.0

        f=np.empty([self.initVars.nVars],'float') #create a 1 x 1 element vector to hold the derivitive
        f[0]= y[0]*(beta_g*x - u.chi)
        return f


if __name__=="__main__":
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
 
    theSolver=Integ52('daisy.ini')
    timeVals,yVals,errorList=theSolver.timeloop5Err()

    thefig=plt.figure(1)
    thefig.clf()
    theAx=thefig.add_subplot(111)
    theLines=theAx.plot(timeVals,yVals)
    theAx.set_title('lab 5 interactive 2')
    theAx.set_xlabel('time')
    theAx.set_ylabel('fractional coverage')
    theAx.legend(theLines,('grey daisies',),loc='best')
    plt.show()
    
