try:
    from lab5_funs import Integrator
except ImportError:
    import os,sys
    libdir=os.path.abspath('../')
    sys.path.append(libdir)
    from lab5_funs import Integrator

class Integ53(Integrator):

    def derivs5(self,y,t):
        """y[0]=fraction white daisies
           y[1]=fraction black daisies
           no feedback between daisies and
           albedo_p (set to ground albedo)
        """
        sigma=5.67e-8  #Stefan Boltzman constant W/m^2/K^4
        u=self.userVars
        x = 1.0 - y[0] - y[1]
        albedo_p = u.albedo_ground
        Te_4 = u.S0/4.0*u.L*(1.0 - albedo_p)/sigma
        eta = u.R*u.S0/(4.0*sigma)
        temp_b = (eta*(albedo_p - u.albedo_black) + Te_4)**0.25
        temp_w = (eta*(albedo_p - u.albedo_white) + Te_4)**0.25

        if(temp_b >= 277.5 and temp_b <= 312.5): 
            beta_b= 1.0 - 0.003265*(295.0 - temp_b)**2.0
        else:
            beta_b=0.0

        if(temp_w >= 277.5  and temp_w <= 312.5): 
            beta_w= 1.0 - 0.003265*(295.0 - temp_w)**2.0
        else:
            beta_w=0.0

        f=np.empty([self.initVars.nVars],'float') #create a 1 x 2 element vector to hold the derivitive
        f[0]= y[0]*(beta_w*x - u.chi)
        f[1] = y[1]*(beta_b*x - u.chi)
        return f



if __name__=="__main__":
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
 
    theSolver=Integ53('daisy.ini')
    timeVals,yVals,errorList=theSolver.timeloop5Err()

    thefig=plt.figure(1)
    thefig.clf()
    theAx=thefig.add_subplot(111)
    theLines=theAx.plot(timeVals,yVals)
    theLines[1].set_linestyle('--')
    theLines[1].set_color('k')
    theAx.set_title('lab 5 interactive 3')
    theAx.set_xlabel('time')
    theAx.set_ylabel('fractional coverage')
    theAx.legend(theLines,('white daisies','black daisies'),loc='center right')
    plt.show()
    
