'''
This module analysis a gaussian information engine. It uses the results of the Information Engine module to calculate output power,
performnace and free-energy change in order to analyse the efficiency of an informatin engine.

This module requires numpy and InformationEngines.

Classes
-------
Create_Data
    runs the information engine simulation and collects its data
Statistics
    creates statistics on data
Analysor
    analyses data of information engine
'''


import InformationEngine as info
import numpy as np


#get data from simulation - depending on information engine 
class Create_Data:
    '''
    a class used to represent data of a simulated information engine
    
    ...
    
    Attributes
    ----------
    length : int
        integer that resembles maximal number of ratcheting events at a single simulation -- default value: 1e4
    frequency : float
        sampling frequency of ratcheting events (positional update of bead and trap) -- default value: 100    
    
    Methods
    -------
    single_run()
        runs one simulation and collects all bead and trap positions
    statistics(amount : int)
        runs multiple simulations and collects all bead and trap positions
    '''


    def __init__(self, sigma: float, alpha: float, threshold: float, psi: float, length: float, frequency:float):
        '''initialises  simulation'''

        self.frequency = frequency
        self.length = length
        self.sigma = sigma
        self.alpha = alpha
        self.threshold = threshold
        self.psi = psi

        #starts simulation
        self.simulation = info.Simulation(length)

        self.positions = []
        self.traps_after = []
        self.traps_before = []


    def single_run(self): 
        '''collects data of single simulation run'''
        initial_bead_position = 0
        initial_trap_position = 0 

        #creating raw data
        self.simulation.single_trajectory(self.frequency, initial_bead_position, initial_trap_position, self.sigma, self.alpha, self.threshold, self.psi)

        self.positions = self.simulation.bead_positions
        self.traps_before = self.simulation.traps_before
        self.traps_after = self.simulation.traps_after
        

    def statistics(self, amount: int):
        '''take statistics of multiple runs'''

        #initial values
        initial_bead_position = 0  
        initial_trap_position = 0 

        #input
        self.amount = amount
        
        for n in range(amount):

            self.simulation.single_trajectory(self.frequency, initial_bead_position, initial_trap_position, self.sigma, self.alpha, self.threshold, self.psi)
            self.positions.append(self.simulation.bead_positions)
            self.traps_before.append(self.simulation.traps_before)
            self.traps_after.append(self.simulation.traps_after)




class Analysor:
    '''
    a class that represents objects to analyse the thermofynamics of a gaussian information engine
    
    ...

    Attributes
    ----------
    data : CreateData
        data from information engine that is to be analysed

    Methods
    -------
    trap_work() 
        calculates the work done by the trap after bead measurement
    free_energy()
        calculates free energy change after applying feedback
    performance()
        calculates performance of information engine
    velocity()
        calculates velocity of bead 
    power()
        calculates power of bead
    '''

    def __init__(self, sigma: float, alpha: float, threshold: float, psi: float, amount=1, length=1000000, frequency=100):
        '''initialises analysis, takes data from CreateData module'''

        #crucial input
        self.sigma = sigma
        self.alpha = alpha

        #initialises
        data = Create_Data(sigma, alpha, threshold, psi, length, frequency)

        data.statistics(amount)

        self.positions = data.positions
        self.traps_after = data.traps_after
        self.traps_before = data.traps_before

        self.frequency = data.frequency
        self.amount = data.amount
        self.length = data.length


    def trap_work(self):
        '''calculates the work done by the trap after bead measurement'''

        work = np.zeros_like(self.positions)

        for index, position in enumerate(self.positions):

            work[index] = 0.5*(np.power((position - self.traps_after[index]),2) - np.power((position - self.traps_before[index]),2))

        return work


    def free_energy(self):
        '''calculates free energy change after applying feedback'''    

        self.bead = info.Bead(0, self.length, 1/self.frequency)
        energy = np.zeros_like(self.positions)

        for index, trap_after in enumerate(self.traps_after):

            energy[index] = self.bead.delta_g*(trap_after - self.traps_before[index])   

        return energy  


    def performance(self):
        '''calculates performance of an engine after applying feedback'''                     
    
        self.trap = info.Trap(0, self.sigma, self.alpha)
        performances = np.zeros_like(self.positions)

        energy = self.free_energy()
        trap_work = self.trap_work()

        for index, work in enumerate(trap_work):

            performances[index] = 2*self.trap.gamma*energy[index] - 2*(1 - self.trap.gamma)*work

        return performances    


    def velocity(self):
        '''calculates velocity'''
        velocities = np.zeros_like(self.positions)

        for index, position in enumerate(self.positions):

            velocities[index] = position*(self.frequency/self.length)

        return velocities    


    def power(self):
        '''calculates power'''

        self.bead = info.Bead(0, self.length, 1/self.frequency)
        powers = np.zeros_like(self.positions)

        velocities = self.velocity()

        for index, velocity in enumerate(velocities):

            powers[index] = velocity*self.bead.delta_g

        return powers    


    def __entropy__(self, traps: np.array):
        '''calculates entropy change'''

        relative_position = np.zeros_like(self.positions)

        for idx, itm in enumerate(self.positions):

            relative_position[idx] =  itm - traps[idx]

    
        counts = []
        bins  = []


        for n in range(len(relative_position)):

            count, bin = np.histogram(relative_position[n], bins=100)

            counts.append(count)
            bins.append(bin)


        results = np.zeros((len(relative_position), 100))

        for i in range(len(counts)):
            for j in range(len(counts[i])):

                bin_size = bins[i][1] - bins[i][0]            

                if counts[i][j] == 0:

                    results[i][j] = 0

                else:    

                    results[i][j] = - counts[i][j]/(self.amount*self.length)* np.log(counts[i][j]/(self.amount*self.length*bin_size)) 

        return results


    def entropy_change(self):
        '''calculates entropy change before and after trap shift'''

        entropy_before = self.__entropy__(self.traps_before)
        entropy_after = self.__entropy__(self.traps_after)

        change = np.zeros((len(entropy_before), len(entropy_before[0])))

        for i in range(len(entropy_after)):
            for j in range(len(entropy_after[i])):

                change[i, j] = entropy_after[i, j] - entropy_before[i, j]

        return change


    def additional_power(self):
        '''calculates total additional work'''

        entropy = self.entropy_change()

        total_entropy = - np.sum(np.mean(entropy, axis=0))*self.frequency

        return total_entropy      


    def free_energy_rate(self):
        '''calculates free-energy rate after feedback application of single simulation run'''

        energy = self.free_energy()

        free_energy = np.sum(energy*self.frequency)/(self.amount*self.length)

        return free_energy 


    def trap_work_rate(self):
        '''calculates trap-work rate of single simulation run''' 

        trap_work = self.trap_work()

        work = np.sum(trap_work*self.frequency)/(self.amount*self.length)

        return work       

        
    def output_power(self):
        '''calculates difference between free-energy change and input trap work'''

        power = self.free_energy_rate() - self.trap_work_rate() - self.additional_power()

        return power        
