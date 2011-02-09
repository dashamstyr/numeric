from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
 

def initinter41(valueDict):
# function to initialize variable for example 1
    class coeff(object):
        pass
    theCoeff=coeff()
    t_beg = valueDict['t_beg']
    t_end = valueDict['t_end']
    theCoeff.dt = valueDict['dt']
    theCoeff.c1 = valueDict['c1']
    theCoeff.c2 = valueDict['c2']
    theCoeff.c3 = valueDict['c3']
    yinitial=1
    return (t_beg,t_end,theCoeff,yinitial)

def derivsinter41(coeff, y, theTime):
    f=coeff.c1*y + coeff.c2*theTime+coeff.c3
    return f

def eulerinter41(coeff,y,theTime):
    y=y + coeff.dt*derivsinter41(coeff,y,theTime)
    return y

def midpointinter41(coeff, y,theTime):
    midy=y + 0.5 * coeff.dt * derivsinter41(coeff,y,theTime)
    y = y + coeff.dt*derivsinter41(coeff,midy,theTime+0.5*coeff.dt)
    return y


def rk4ODEinter41(coeff, y, theTime):
  k1 = coeff.dt * derivsinter41(coeff,y,theTime)
  k2 = coeff.dt * derivsinter41(coeff,y + (0.5 * k1),theTime+0.5*coeff.dt)
  k3 = coeff.dt * derivsinter41(coeff,y + (0.5 * k2),theTime+0.5*coeff.dt)
  k4 = coeff.dt * derivsinter41(coeff,y +  k3,theTime+coeff.dt)
  y = y + (1.0/6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4)
  return y

def rkckODEinter41(coeff,yold,told):

## initialize the Cash-Karp coefficients
## defined in the tableau in lab 4,
## section 3.5

  a = [.2, 0.3, 0.6, 1.0, 0.875]
  c1 = [37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0]
  c2= [2825.0/27648.0, 0.0, 18575.0/48384.0, 13525.0/55296.0,
       277.0/14336.0, .25]
  b=np.empty([5,5],'float')
  c2 = c1 - c2
  b[0,0] =0.2 
  b[1,0]= 3.0/40.0 
  b[1,1]=9.0/40.0
  b[2,0]=0.3 
  b[2,1]=-0.9 
  b[2,2]=1.2
  b[3,0]=-11.0/54.0 
  b[3,1]=2.5 
  b[3,2]=-70.0/27.0 
  b[3,3]=35.0/27.0
  b[4,0]=1631.0/55296.0 
  b[4,1]=175.0/512.0 
  b[4,2]=575.0/13824.0
  b[4,3]=44275.0/110592.0 
  b[4,4]=253.0/4096.0

# set up arrays
  
  derivArray=np.empty([6,length(yold)],'float')
  ynext=np.empty_like(yold)
  bsum=np.empty_like(yold)
  derivArray[0,:]=derivsinter41(coeff,yold,told)
  
# calculate step
  
  y=yold
  for i in np.arange(5):
    bsum=0.
    for j in np.arange(i):
      bsum=bsum + b[i,j]*derivArray[j,:]
    derivArray[i+1,:]=derivsinter41(coeff,y + coeff.dt*bsum,told + a[i]*coeff.dt)
    ynext = ynext + c1[i]*derivArray[i,:]
  y = y + coeff.dt*(ynext + c1[6]*derivArray[5,:])
  

if __name__=="__main__":
    initialVals={'t_beg':0.,'t_end':1.,'dt':0.05,'c1':-1.,'c2':1.,'c3':1.}
    t_beg,t_end,coeff,yinitial = initinter41(initialVals)
    timeVec=np.arange(t_beg,t_end,coeff.dt)
    nsteps=len(timeVec)
    ye=[]
    ym=[]
    y=yinitial
    ye.append(yinitial)
    ym.append(yinitial)
    for i in np.arange(1,nsteps):
        ynew=eulerinter41(coeff,y,timeVec[i-1])
        ye.append(ynew)
        ynew=midpointinter41(coeff,y,timeVec[i-1])
        ym.append(ynew)
        y=ynew
    analytic=timeVec + np.exp(-timeVec)
    theFig=plt.figure(0)
    theFig.clf()
    theAx=theFig.add_subplot(111)
    l1=theAx.plot(timeVec,analytic,'b-')
    theAx.set_xlabel('time (seconds)')
    l2=theAx.plot(timeVec,ye,'r-')
    l3=theAx.plot(timeVec,ym,'g-')
    theAx.legend((l1,l2,l3),('analytic','euler','midpoint'),
                 loc='best')
    plt.show()



## % load the input data 
## function interative42
## [t_beg,t_end,coeff.dt,coeff.c1,coeff.c2,coeff.c3,y(1),y(2)] = initinter41

## time=t_beg:coeff.dt:t_end  %create the time vector
## time=time'                 %transpose it from [1, nsteps] to [nsteps, 1]
##                             %we need to do this because gplot plots
## 			    %columns, not rows
## [nsteps,done]=size(time) % done is an un used variable

## if(nsteps <= 2)             %here's and example of rudimentary error checking
##   error('need at least two steps')
## end

## %create a column vector to hold y at each timestep
## %for later plotting

## savedatae=zeros([nsteps,1])
## savedatam=zeros([nsteps,1])
## savedater=zeros([nsteps,1])
## ye = y
## ym = y
## yr = y
## savedatae[1)=ye[1)
## savedatam[1)=ym[1)
## savedatar[1)=yr[1)

## for i=2:nsteps
##   ye=eulerinter41(coeff,ye,time[i-1))
##   savedatae[i)=ye[1)
##   ym=midpointinter41(coeff,ym,time[i-1))
##   savedatam[i)=ym[1)
##   yr=rk4ODEinter41(coeff,yr,time[i-1))
##   savedatar[i)=yr[1)
## end

## dt = (t_end-t_beg)/99
## for i=1:100
##     timea[i) = t_beg + dt*(i-1)
##     savedataa[i)= timea[i) + exp(-timea[i))
## end

## hold off
## plot (time,savedatae,'Blue'), xlabel 'time (seconds)', ylabel 'y1'
## hold on
## plot (time,savedatam,'Red')
## plot (time,savedatar,'Magenta')
## plot (timea,savedataa,'Green')



## % load the input data 
## function interative43
## [t_beg,t_end,coeff.dt,coeff.c1,coeff.c2,coeff.c3,y(1),y(2)] = initinter41;

## time=t_beg:coeff.dt:t_end;  %create the time vector
## time=time';                 %transpose it from [1, nsteps] to [nsteps, 1]
##                             %we need to do this because gplot plots
## 			    %columns, not rows
## [nsteps,done]=size(time); % done is an un used variable

## if(nsteps <= 2)             %here's and example of rudimentary error checking
##   error('need at least two steps');
## end

## %create a column vector to hold y at each timestep
## %for later plotting

## savedatar=zeros([nsteps,1]);
## savedater5=zeros([nsteps,1]);
## yr = y;
## yr5 = y;
## savedatar(1)=yr(1);
## savedatar5(1)=yr5(1);

## for i=2:nsteps
##   yr=rk4ODEinter41(coeff,yr,time(i-1));
##   savedatar(i)=yr(1);
##   yr5=rkckODEinter41(coeff,yr5,time(i-1));
##   savedatar5(i)=yr5(1);
## end

## dt = (t_end-t_beg)/99;
## for i=1:100
##     timea(i) = t_beg + dt*(i-1);
##     savedataa(i)= timea(i) + exp(-timea(i));
## end

## hold off
## plot (time,savedatar5,'Black')
## hold on
## plot (time,savedatar,'Magenta')
## plot (timea,savedataa,'Green')




