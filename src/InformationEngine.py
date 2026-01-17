'''
This module creates simulations of an information engine. It has a Trap and a Bead class, creating the bead behaviour
(which in this case is the Langevin dynamics) and the feedback rules for the trap. The simulation class simulates bead and trap 
movement according to a certain sampling time. The sampling time is the time at which the bead is updated (i.e. the 
time the bead has free movement until its position measured). In this project we are interested to look at certain sampling 
frequencies; that is the frequency at which the bead position is measured.

This module requires numpy.

Classes
-------
Trap 
    represents a harmonic potential in which the bead is trapped
Bead
    represents a bead with mass, etc (given by scaled unit delta_g) that moves under Langevin dynamics
Simulation
    simulates bead and trap movement under certain Feedback rules

'''


import numpy as np
import math

class Trap:
    '''represents a hamonic potential (i.e. bead trap)'''

    def __init__(self, position, sigma: float, alpha: float, threshold: float, psi: float, length: float, gamma=0.5):
        '''initialises trap position'''

        self.position = position
        self.gamma = gamma
        self.sigma = sigma
        self.alpha = alpha
        self.threshold = threshold
        self.psi = psi

        self.length = length

        self.randomness = np.random.normal(size=length)


    def position_update(self, bead_position: float, step: int):
        '''attribute that moves the trap according to bead position'''

        #measurement error
        bead_position_estimate = bead_position + self.sigma*self.randomness[step]

        relative_bead_position = bead_position_estimate - self.position

        #position update
        new_trap_position = self.position + self.alpha*relative_bead_position + self.psi

        #update rule
        if relative_bead_position - self.threshold > 0:

            self.position = new_trap_position


    def position_update_no_threshold(self, bead_position: float, alpha=1):
        '''moves trap according to bead position without threshold'''

        self.bead = Bead(0, self.length)

        relative_bead_position = bead_position - self.position

        #position update
        psi = alpha*(self.bead.delta_g/(2*(1 - self.gamma)))
        new_trap_position = self.position + alpha*relative_bead_position + psi

        #update rule -> no-threshold, therefore always updated
        self.position = new_trap_position



class Bead:
    '''creates particle with position and determines movement'''


    def __init__(self, position: float, length: int, sampling_time: float, delta_g=0.84):
        '''initialises the particle position'''

        self.delta_g = delta_g
        self.position = position
        self.exponential_time = math.exp(-sampling_time)
        self.randomness = np.random.normal(size=length)


    def deterministic_update(self, trap_location: float):
        '''determines deterministic particle movement under langevin dynamics'''

        self.position = self.exponential_time*self.position + (1 - self.exponential_time)*(trap_location - self.delta_g)


    def position_update(self, trap_location: float, step: int):
        '''determmines particle movement under a given timestep'''

        variance = (1 - self.exponential_time**2)**(1/2)
        fluctuation = self.randomness[step]

        stochastic_part = variance*fluctuation

        self.deterministic_update(trap_location)
        self.position += stochastic_part
 

class Simulation:
    '''represents simulations of trap and position updates'''

    def __init__(self, length: float):
        '''initialises simulation start, trap and bead are at position 0'''

        self.length = int(length)


    def single_trajectory(self, frequency: float, b_position: float, t_position: float, sigma: float, alpha: float, threshold: float, psi: float):
        '''sets bead and trap to new positions'''

        #initial values
        b = Bead(b_position, self.length, 1/frequency)
        t = Trap(t_position, sigma, alpha, threshold, psi, self.length)

        steps = 0
        ignored_steps = int(self.length*0.01)

        self.bead_positions = np.zeros(self.length - ignored_steps)
        self.traps_after = np.zeros(self.length - ignored_steps)
        self.traps_before = np.zeros(self.length- ignored_steps)


        while (steps < ignored_steps):

            b.position_update(t.position, steps)
            t.position_update(b.position, steps)

            steps = steps + 1
        

        while (steps < self.length):

            self.traps_before[steps - ignored_steps] = t.position

            b.position_update(t.position, steps)
            t.position_update(b.position, steps)

            self.bead_positions[steps - ignored_steps] = b.position
            self.traps_after[steps - ignored_steps] = t.position

            steps = steps + 1         
            

    def multiple_trajectories(self, frequency: np.array):
        '''sets bead and trap to new positions a couple of times''' 

        self.final_positions = np.zeros_like(frequency)

        b_position = 0
        t_position = 0

        for index, value in enumerate(frequency):

            self.single_trajectory(value, b_position, t_position, sigma=0, alpha=2, threshold=0, psi=0)
            self.final_positions[index] = self.bead_positions[-1]
            b_position = self.bead_positions[-1] - self.traps_after[-1]
