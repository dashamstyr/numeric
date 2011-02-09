try:
    from lab5_funs import Integrator, HoldVals, rkck_init
except ImportError:
    import os,sys
    libdir=os.path.abspath('../')
    sys.path.append(libdir)
    from lab5_funs import Integrator

import ConfigParser

class Integ55(Integrator):
    """rewrite the init and derivs5 methods to
       work with the exponential equation
    """

    def __init__(self,coeffFileName):

        Integrator.__init__(self,coeffFileName)
        i=self.initVars
        i.yinit=np.array([i.yval])
        i.nVars=len(i.yinit)

    def derivs5(self,y,theTime):
        """y[0]=fraction white daisies
        """
        u=self.userVars
        f=np.empty_like(self.initVars.yinit)
        f[0]=u.c1*y[0] + u.c2*theTime + u.c3;
        return f


if __name__=="__main__":
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt

    theSolver=Integ55('expon.ini')

    timeVals,yVals,yErrors =theSolver.timeloop5Err()
    timeVals=np.array(timeVals)
    exact=timeVals + np.exp(-timeVals)
    yVals=np.array(yVals)
    yVals=yVals.squeeze()
    yErrors=np.array(yErrors)

    thefig=plt.figure(1)
    thefig.clf()
    theAx=thefig.add_subplot(111)
    line1=theAx.plot(timeVals,yVals)
    line2=theAx.plot(timeVals,exact,'r+')
    theAx.set_title('lab 5 interactive 5')
    theAx.set_xlabel('time')
    theAx.set_ylabel('y value')
    theAx.legend((line1,line2),('adapt','exact'),loc='center right')

    #
    # we need to unpack yvals (a list of arrays of length 1
    # into an array of numbers using a list comprehension
    #
    
    thefig=plt.figure(2)
    thefig.clf()
    theAx=thefig.add_subplot(111)
    realestError = yVals - exact
    actualErrorLine=theAx.plot(timeVals,realestError)
    estimatedErrorLine=theAx.plot(timeVals,yErrors)
    theAx.legend((actualErrorLine,estimatedErrorLine),\
                 ('actual error','estimatedError'),loc='best')


    timeVals,yVals,yErrors =theSolver.timeloop5fixed()

    np_yVals=np.array(yVals).squeeze()
    yErrors=np.array(yErrors)
    np_exact=timeVals + np.exp(-timeVals)

    
    thefig=plt.figure(3)
    thefig.clf()
    theAx=thefig.add_subplot(111)
    line1=theAx.plot(timeVals,np_yVals)
    line2=theAx.plot(timeVals,np_exact,'r+')
    theAx.set_title('lab 5 interactive 5 -- fixed')
    theAx.set_xlabel('time')
    theAx.set_ylabel('y value')
    theAx.legend((line1,line2),('fixed','exact'),loc='center right')

    thefig=plt.figure(4)
    thefig.clf()
    theAx=thefig.add_subplot(111)
    realestError = np_yVals - np_exact
    actualErrorLine=theAx.plot(timeVals,realestError)
    ## estimatedErrorLine=theAx.plot(timeVals,yErrors)
    ## theAx.legend((actualErrorLine,estimatedErrorLine),\
    ##              ('actual error','estimatedError'),loc='best')
    ## theAx.set_title('lab 5 interactive 5 -- fixed errors')

    plt.show()
