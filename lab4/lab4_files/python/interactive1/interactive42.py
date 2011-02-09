from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from functions4 import initinter41,eulerinter41,midpointinter41,rk4ODEinter41


if __name__=="__main__":
    initialVals={'t_beg':0.,'t_end':1.,'dt':0.05,'c1':-1.,'c2':1.,'c3':1.}
    t_beg,t_end,coeff,yinitial = initinter41(initialVals)
    timeVec=np.arange(t_beg,t_end,coeff.dt)
    nsteps=len(timeVec)
    ye=[]
    ym=[]
    yrk=[]
    y=yinitial
    ye.append(yinitial)
    ym.append(yinitial)
    yrk.append(yinitial)
    for i in np.arange(1,nsteps):
        ynew=eulerinter41(coeff,y,timeVec[i-1])
        ye.append(ynew)
        ynew=midpointinter41(coeff,y,timeVec[i-1])
        ym.append(ynew)
        ynew=rk4ODEinter41(coeff,y,timeVec[i-1])
        yrk.append(ynew)
        y=ynew
    analytic=timeVec + np.exp(-timeVec)
    theFig=plt.figure(0)
    theFig.clf()
    theAx=theFig.add_subplot(111)
    l1=theAx.plot(timeVec,analytic,'b-')
    theAx.set_xlabel('time (seconds)')
    l2=theAx.plot(timeVec,ye,'r-')
    l3=theAx.plot(timeVec,ym,'g-')
    l4=theAx.plot(timeVec,yrk,'m-')
    theAx.legend((l1,l2,l3,l4),('analytic','euler','midpoint','rk'),
                 loc='best')
    theAx.set_title('interactive 4.2')
    plt.show()
