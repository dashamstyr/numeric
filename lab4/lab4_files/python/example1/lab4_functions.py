import numpy as np
import matplotlib.pyplot as plt

def derivs4(coeff, y):
  f=np.empty_like(y) #create a 2 element vector to hold the derivitive
  f[0]=y[1]
  f[1]= -1.*coeff.c1*y[1] - coeff.c2*y[0]
  return f

def euler4(coeff,y):
  ynew=y + coeff.dt*derivs4(coeff,y)
  return ynew

def init2():
# function to initialize variable for example 1
    class coeff(object):
        pass
    theCoeff=coeff()
    t_beg = 0.
    t_end = 40.
    theCoeff.dt = 0.1
    theCoeff.c1 = 0.
    theCoeff.c2 = 1.
    y=np.array([0,1],'float')
    return (t_beg,t_end,theCoeff,y)

def midpoint4(coeff, y):
  ynew = y + coeff.dt*derivs4(coeff,y + (0.5 * coeff.dt * derivs4(coeff,y)))
  return ynew


def rk4ODE(coeff, y):
  k1 = coeff.dt * derivs4(coeff,y)
  k2 = coeff.dt * derivs4(coeff,y + (0.5 * k1))
  k3 = coeff.dt * derivs4(coeff,y + (0.5 * k2))
  k4 = coeff.dt * derivs4(coeff,y +  k3)
  ynew = y + (1.0/6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4)
  return ynew

if __name__ == "__main__":
    t_beg,t_end,coeff,y=init2()
    time=np.arange(t_beg,t_end,coeff.dt)
    nsteps=len(time) 
    savedata=np.empty([nsteps],'float')
    for i in range(nsteps):
        y=midpoint4(coeff,y)
        savedata[i]=y[0]

    theFig=plt.figure(0)
    theFig.clf()
    theAx=theFig.add_subplot(111)
    theAx.plot(time,savedata,'o-')
    theAx.set_title('y1')
    theAx.set_xlabel('time (seconds)')
    theAx.set_ylabel('y12')

    plt.show()




