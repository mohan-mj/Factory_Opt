# -*- coding: utf-8 -*-

import numpy as np # General numerics
from scipy.integrate import odeint # Integration
from scipy.optimize import minimize # Optimization
import matplotlib.pyplot as plt # Plotting

class SNOWBALL_OPT():

    def __init__(self):
        # Define System Parameters
        self.K0 = 85 # Snowball growth factor 1
        self.beta = 0.07 # Snowball growth factor 2
        self.C_d = 0.3 # Drag coefficient
        self.g = 9.8 # Gravity
        self.rho = 350 # Snow density
        self.theta = np.radians(5) # Slope
        self.rho_a = 0.9 # Air density
        self.p = [self.K0,self.C_d,self.g,self.rho,self.theta,self.rho_a,self.beta]

        # Target force
        self.F_d = 25000

        # Initial Snowball Conditions
        m0 = 10 # Initial mass
        self.v0 = 0 # Initial velocity
        # self.r0 =  __radius_from_mass(m0) # Initial radius
        self.s0 = 0 # Initial position


    def radius_from_mass(self, m):
        return (m/(4/3.0*np.pi*self.rho))**(1/3.0)


    def snowball_dynamics(self, w, t):
        """    
        This function defines the dynamics of our snowball, the equations of motion
        and the rate at which it changes size and mass."""
        # unpack state variables
        M,r,s,v = w
        
        # unpack parameters
        # K0,C_d,g,rho,theta,rho_a,beta = p
        
        # Make an array of the right hand sides of the four differential equations that make up our system.
        f = [self.beta * self.K0 * np.exp(-self.beta*t),
            (self.beta * self.K0 * np.exp(-self.beta*t))/(4*np.pi*self.rho*r**2),
            v,
            (-15*self.rho_a*self.C_d)/(56*self.rho)*1/r*v**2 - \
                23/7*1/r*self.beta*self.K0*np.exp(-self.beta*t)/ \
                    (4*np.pi*self.rho*r**2)*v+5/7*self.g*np.sin(self.theta)]
        return f


    def objective(self, m0):
        """This is the objective function of our optimization.  The optimizer will attempt
        to minimize the output of this function by changing the initial snowball mass. """   
        # Load parameters
        # 
        
        # Get initial radius from initial mass
        r0 = self.radius_from_mass(m0) #(m0/(4/3.0*np.pi*self.rho))**(1/3.0)

        # Set initial guesses
        guess = [m0,r0,self.s0,self.v0]

        # Set up time array to solve for 30 seconds
        t = np.linspace(0,30)

        # Integrate forward for 30 seconds
        sol = odeint(self.snowball_dynamics, guess, t)
        
        # Calculate kinetic energy at the end of the run
        ke = 0.5 * sol[:,0][-1] * sol[:,3][-1]**2

        # Calculate force required to stop snowball in one snowball radius
        F = ke / sol[:,1][-1]
        
        # Compare to desired force : This should equal zero when we are done
        obj = (F - self.F_d)**2
        
        return obj

if __name__=="__main__":

    Snow = SNOWBALL_OPT()
    # Call optimization using the functions defined above
    res = minimize(Snow.objective, 10, options={'disp':True})    

    # Get optimized initial mass from solution
    m0_opt = res.x[0]

    # Calculate optimized initial radius from initial mass
    r0_opt = Snow.radius_from_mass(m0_opt)

    print('Initial Mass: ' + str(m0_opt) + ' kg (' + str(m0_opt*2.02) + ' lbs)')
    print('Initial Radius: ' + str(r0_opt*100) + ' cm (' + str(r0_opt*39.37) + ' inches)')



