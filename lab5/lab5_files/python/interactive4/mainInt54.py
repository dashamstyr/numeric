try:
    from lab5_funs import Integrator
except ImportError:
    import os,sys
    libdir=os.path.abspath('../')
    sys.path.append(libdir)
    from lab5_funs import Integrator

if __name__=="__main__":
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
 
    theSolver=Integrator('daisy.ini')
    timeVals,yVals,errorList = theSolver.timeloop5Err()

    thefig=plt.figure(1)
    thefig.clf()
    theAx=thefig.add_subplot(111)
    theLines=theAx.plot(timeVals,yVals)
    theLines[1].set_linestyle('--')
    theLines[1].set_color('k')
    theAx.set_title('lab 5 interactive 4')
    theAx.set_xlabel('time')
    theAx.set_ylabel('fractional coverage')
    theAx.legend(theLines,('white daisies','black daisies'),loc='center right')
    plt.show()
    
