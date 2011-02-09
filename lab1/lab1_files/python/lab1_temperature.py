#!/usr/bin/env python
"""
Script to plot the exponential decay of temperature of four objects
with different initial temperature in a single ambient temperature.

Example usage from python: ::

     >> import lab1_temperature

Example usage from ipython: ::

     >> run lab1_temperature

"""
from numpy import array, arange, exp
import  matplotlib.pyplot as plt


# set the time scale (seconds)
t = arange(0., 400000., 100.)

# set the ambient temperature (Celcius)
Ta = 20.

# set the four initial temperatures (Celcius)
To = array([-10., 10., 20., 30.])

# set the time constant of equilibriation (1/seconds)
la = 0.00001

# calculate the temperatures with time of the four objects
a = Ta + (To[0] - Ta) * exp(-la * t)
b = Ta + (To[1] - Ta) * exp(-la * t)
c = Ta + (To[2] - Ta) * exp(-la * t)
d = Ta + (To[3] - Ta) * exp(-la * t)

# plot the tempeatures
fig=plt.figure(1)
fig.clf()
t=t/3600.
plt.plot(t, a)
plt.plot(t, b)
plt.plot(t, c)
plt.plot(t, d)
plt.xlabel('time (hours)')
plt.ylabel('temperature (deg C)')

# add a legend & show the graph
plt.legend(["To = %s" % To[0], 
        "To = %s" % To[1], 
        "To = %s" % To[2],
        "To = %s" % To[3]], loc="lower right")

plt.show()
