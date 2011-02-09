try:
    from lab5_funs import Integrator
except ImportError:
    import os,sys
    libdir=os.path.abspath('../')
    sys.path.append(libdir)
    from lab5_funs import Integrator

class Integ51(Integrator):
    def derivs5(self,y,t):
        """y[0]=fraction white daisies
           y[1]=fraction black daisies
        """
        sigma=5.67e-8  #Stefan Boltzman constant W/m^2/K^4
        u=self.userVars
        x = 1.0 - y[0] - y[1]        
        albedo_p = x*u.albedo_ground + y[0]*u.albedo_white + y[1]*u.albedo_black    
        Te_4 = u.S0/4.0*u.L*(1.0 - albedo_p)/sigma
        eta = u.R*u.S0/(4.0*sigma)
        temp_b = (eta*(albedo_p - u.albedo_black) + Te_4)**0.25
        temp_w = (eta*(albedo_p - u.albedo_white) + Te_4)**0.25

        #growth rates don't depend on temperature
        beta_b = 0.1 # growth rate for black daisies
        beta_w = 0.7 # growth rate for white daisies

        f=np.empty([self.initVars.nVars],'float') #create a 1 x 2 element vector to hold the derivitive
        f[0]= y[0]*(beta_w*x - u.chi)
        f[1] = y[1]*(beta_b*x - u.chi)
        return f



if __name__=="__main__":
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
 
    theSolver=Integ51('daisy.ini')
    timeVals,yVals,errorList=theSolver.timeloop5Err()

    thefig=plt.figure(1)
    thefig.clf()
    theAx=thefig.add_subplot(111)
    theLines=theAx.plot(timeVals,yVals)
    theLines[1].set_linestyle('--')
    theLines[1].set_color('k')
    theAx.set_title('lab 5 interactive 1')
    theAx.set_xlabel('time')
    theAx.set_ylabel('fractional coverage')
    theAx.legend(theLines,('white daisies','black daisies'),loc='best')


    thefig=plt.figure(2)
    thefig.clf()
    theAx=thefig.add_subplot(111)
    theLines=theAx.plot(timeVals,errorList)
    theLines[1].set_linestyle('--')
    theLines[1].set_color('k')
    theLines[0].set_marker('+')
    theLines[1].set_marker('o')
    theAx.set_title('lab 5 interactive 1 errors')
    theAx.set_xlabel('time')
    theAx.set_ylabel('errors')
    theAx.legend(theLines,('white errors','black errors'),loc='best')
    thefig.canvas.draw()

    timeVals,yVals,errorList=theSolver.timeloop5fixed()

    thefig=plt.figure(3)
    thefig.clf()
    theAx=thefig.add_subplot(111)
    theLines=theAx.plot(timeVals,yVals)
    theLines[0].set_marker('+')
    theLines[1].set_linestyle('--')
    theLines[1].set_color('k')
    theLines[1].set_marker('*')
    theAx.set_title('lab 5 interactive 1 fixed')
    theAx.set_xlabel('time')
    theAx.set_ylabel('fractional coverage')
    theAx.legend(theLines,('white daisies','black daisies'),loc='best')

    thefig=plt.figure(4)
    thefig.clf()
    theAx=thefig.add_subplot(111)
    theLines=theAx.plot(timeVals,errorList)
    theLines[0].set_marker('+')
    theLines[1].set_linestyle('--')
    theLines[1].set_color('k')
    theLines[1].set_marker('*')
    theAx.set_title('lab 5 interactive 1 fixed errors')
    theAx.set_xlabel('time')
    theAx.set_ylabel('fractional coverage')
    theAx.legend(theLines,('white errors','black errors'),loc='best')


    plt.show()
    
