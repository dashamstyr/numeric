from __future__ import division
from ConfigParser import ConfigParser
import sys,os
import numpy as np
import pdb

class HoldVals(object):

    def __init__(self,configList):
        for key,value in configList:
            #
            # convert to a float if possible
            # save the raw value if not
            #
            try:
                self.__setattr__(key,float(value))
            except ValueError:
                self.__setattr__(key,value)
        
    def __str__(self):
        out=''
        for name,value in self.__dict__.iteritems():
            newLine="%s = %s\n" % (name,value)
            out=out+newLine
        return out
    

def rkck_init():
## %
## % initialize the Cash-Karp coefficients
## % defined in the tableau in lab 4,
## % section 3.5
## %
  a = np.array([0.2, 0.3, 0.6, 1.0, 0.875])
  #c1 coefficients for the fifth order scheme
  c1 = np.array([37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0])
  #c2=c* coefficients for the fourth order schme
  c2= np.array([2825.0/27648.0, 0.0, 18575.0/48384.0, 13525.0/55296.0, 277.0/14336.0, .25])
  b=np.empty([5,5],'float')
  #the following line is ci* - ci in lab4, equation 3.12
  c2 = c1 - c2
  #this is the tableu given on lab4, page 7
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
  return (a,c1,c2,b)

class MyConfigParser(ConfigParser):
    optionxform = str

class Integrator(object):
    def __init__(self,coeffFileName):
        print "debug:  Inside Integrator III"
        config = MyConfigParser()
        #
        #preserve case of config variable names
        #
        config.optionxform = str 
        config.read(coeffFileName)
        self.configVals=config
        self.userVars=HoldVals(config.items('userVars'))
        self.timeVars=HoldVals(config.items('timeVars'))
        self.adaptVars=HoldVals(config.items('adaptVars'))
        self.initVars=HoldVals(config.items('initVars'))
        self.rkckConsts=rkck_init()
        #
        # if either whiteConc or blackConc don't
        # exist for this problem, leave them to be
        # set after creation
        #
        i=self.initVars
        try:
            i.yinit=np.array([i.whiteConc,i.blackConc])
            i.nVars=len(i.yinit)
        except AttributeError:
            i.yinit='tbd'
            i.nVars='tbd'

    def __str__(self):
        out='integrator instance with attributes initVars, timeVars,userVars, ' + \
             'adaptVars'
        return out
        
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

    def rkckODE5(self,yold,timeStep,deltaT):

    ## initialize the Cash-Karp coefficients
    ## defined in the tableau in lab 4,
    ## section 3.5

        a,c1,c2,b=self.rkckConsts
        t=self.timeVars
        i=self.initVars
        # set up array to hold k values in lab4 (3.9)
        derivArray=np.empty([6,i.nVars],'float')
        ynext=np.zeros_like(yold)
        bsum=np.zeros_like(yold)
        estError=np.zeros_like(yold)
        #vector k1 in lab4 equation 3.9
        derivArray[0,:]=self.derivs5(yold,timeStep)[:]

        # calculate step
        # c1=c_i in lab 4 (3.9), but c2=c_i - c^*_i

        y=yold
        for i in np.arange(5):
            bsum=0.
            for j in np.arange(i+1):
              bsum=bsum + b[i,j]*derivArray[j,:]
            #vectors k2 through k6 in lab4 equation 3.9
#           pdb.set_trace()
            derivArray[i+1,:]=self.derivs5(y + deltaT*bsum,timeStep + a[i]*deltaT)[:]
            #partial sum of error in lab4 (3.12)
            estError = estError + c2[i]*derivArray[i,:]
            #print "estError: ",estError
            ynext = ynext + c1[i]*derivArray[i,:]
        #final fifth order anser
        y = y + deltaT*(ynext + c1[5]*derivArray[5,:])
        #final error estimate
        estError =  deltaT*(estError + c2[5]*derivArray[5,:])
        #print "estError final: ",estError
        timeStep=timeStep + deltaT
#       pdb.set_trace()
        return (y,estError,timeStep)

    def timeloop5Err(self):
        """return errors as well as values
        """
        t=self.timeVars
        a=self.adaptVars
        i=self.initVars
        nVars=i.nVars
        oldTime=t.tStart
        olddt=t.dt
        yold=i.yinit
        yerror=np.zeros_like(yold)
        num=0
        badsteps=0
        goodsteps=0
        timeVals=[]
        yVals=[]
        errorList=[]
        while(oldTime < t.tEnd):
            timeVals.append(oldTime)
            yVals.append(yold)
            errorList.append(yerror)
            if(num > a.maxSteps):
              raise Exception('num > maxSteps')
            # start out with goodstep false and
            # try different sizes for the next step
            # until one meets the error conditions
            # then move onto next step by setting
            # goodstep to true  
            goodStep=False
            failSteps=0
            while(not goodStep):
                # to exit this loop, need to
                # get the estimated error smaller than
                # the desired error set by the relative
                # tolerance
                if(failSteps > a.maxFail):
                    raise Exception('failSteps > a.maxFail')
                #
                # try a timestep, we may need to reverse this
                #
                ynew,yerror,timeStep=self.rkckODE5(yold,oldTime,olddt)
                print "try a step: : ",ynew
                #
                # lab 5 section 4.2.3
                # find the desired tolerance by multiplying the relative
                # tolerance (RTOL) times the value of y
                # compare this to the error estimate returnd from rkckODE5
                # ATOL takes care of the possibility that y~0 at some point
                #
                errtest=0.
                for i in range(nVars):
                  errtest = errtest + (yerror[i]/(a.ATOL + a.RTOL*np.abs(ynew[i])))**2.0
                errtest = np.sqrt(errtest / nVars)
                #
                # lab5 equation 4.13, S 
                #
                dtChange = a.S*(1.0/errtest)**0.2
                print "dtChange, errtest, timeStep: ",dtChange,errtest,timeStep,ynew,yerror
                if (errtest > 1.0):
                    #estimated error is too big so
                    #reduce the timestep and retry
                    #dtFailMax ~ 0.5, which guarantees that
                    #the new timestep is reduced by at least a
                    #factor of 2
                    #dtFailMin~0.1, which means that we don't trust
                    #the estimate to reduce the timestep by more
                    #than a factor of 10 in one loop
                    if(dtChange > a.dtFailMax):
                        olddt = a.dtFailMax * olddt
                    elif (dtChange < a.dtFailMin): 
                        olddt = a.dtFailMin * olddt
                    else: 
                        olddt = dtChange * olddt
                    if (timeStep + olddt == timeStep):
                        raise Exception('step smaller than machine precision')
                    failSteps=failSteps + 1
                    #
                    # undo the timestep since the error wasn't small enough
                    #
                    ynew = yold
                    timeStep=oldTime
                    #go back to top and see if this olddt produices
                    #a better yerrror
                else:
                    #errtest < 1, so we're happy
                    #try to enlarge the timestep by a factor of dtChange > 1
                    #but keep it smaller than dtPassMax  
                    #try enlarging the timestep bigger for next time
                    #dtpassmin ~ 0.1 and dtpassmax ~ 5
                    if (abs((1.0 - dtChange)) > a.dtPassMin):
                      if(dtChange > a.dtPassMax):
                        dtNew = a.dtPassMax * olddt
                      else:
                        dtNew = dtChange * olddt
                    else:
                      #don't bother changing the step size if
                      #the change is less than dtPassMin
                      dtNew = olddt
                    goodStep = True
                    #
                    # overwrite the old timestep with the new one
                    #
                    oldTime = timeStep
                    yold = ynew
                    #go back up to top while(timeStep < t.tEnd)
                    goodsteps=goodsteps + 1
                #
                # this is number of times we decreased the step size without
                #  advancing
                #
                badsteps=badsteps + failSteps
            #special case if we're within one ortwo timesteps of the end
            #otherwise, set dt to the new timestep size
            if(timeStep + dtNew > t.tEnd):
                olddt = t.tEnd - timeStep
            elif(timeStep + 2.0*dtNew > t.tEnd): 
                olddt = (t.tEnd - timeStep)/2.0
            else:
                olddt = dtNew
        timeVals=np.array(timeVals).squeeze()
        yVals=np.array(yVals).squeeze()
        errorVals=np.array(errorList).squeeze()
        return (timeVals,yVals,errorList)

    def timeloop5fixed(self):
        """fixed time step with
           estimated errors
        """
        t=self.timeVars
        i=self.initVars
        yold=i.yinit
        yError=np.zeros_like(yold)
        yVals=[yold]
        errorList=[yError]
        timeSteps=np.arange(t.tStart,t.tEnd,t.dt)
        for theTime in timeSteps[:-1]:
            yold,yError,newTime=self.rkckODE5(yold,theTime,t.dt)
            yVals.append(yold)
            errorList.append(yError)
        yVals=np.array(yVals).squeeze()
        errorVals=np.array(errorList).squeeze()
        return (timeSteps,yVals,errorVals)



if __name__=="__main__":
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt

    theSolver=Integrator('example1/daisy.ini')

    timeVals,yVals,errList=theSolver.timeloop5Err()
    whiteDaisies=[frac[0] for frac in yVals]

    thefig=plt.figure(1)
    thefig.clf()
    theAx=thefig.add_subplot(111)
    points=theAx.plot(timeVals,whiteDaisies,'b+')
    theLines=theAx.plot(timeVals,yVals)
    theLines[1].set_linestyle('--')
    theLines[1].set_color('k')
    theAx.set_title('lab 5 example 1')
    theAx.set_xlabel('time')
    theAx.set_ylabel('fractional coverage')
    theAx.legend(theLines,('white daisies','black daisies'),loc='best')
    plt.show()
    
