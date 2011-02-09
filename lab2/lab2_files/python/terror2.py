import matplotlib.pyplot as plt
from lab2_functions import euler,leapfrog,runge
import numpy as np

theFuncs={'euler':euler,'leapfrog':leapfrog,'runge':runge}

if __name__=="__main__":
    Ta= 20
    To= 30
    tend = 30.0
    theLambda=-0.8
    npts=10000.
    funChoice='leapfrog'
    #
    #find the method in the theFuncs dictionary and call it
    #
    approxTime,approxTemp=theFuncs[funChoice](npts,tend,To,Ta,theLambda)
    fig1=plt.figure(1)
    fig1.clf()
    plt.plot(approxTime,approxTemp)
    plt.hold(True)
    exactTime=np.empty([npts,],np.float)
    exactTemp=np.empty_like(exactTime)
    for i in np.arange(0,npts):
        exactTime[i] = tend*(i-1)/npts
        exactTemp[i] = Ta + (To-Ta)*np.exp(theLambda*exactTime[i])
    plt.plot(exactTime,exactTemp,'r+')
    plt.hold(False)
    fig2=plt.figure(2)
    fig2.clf()
    difference = approxTemp-exactTemp;
    plt.plot(exactTime,difference)
    plt.show()
