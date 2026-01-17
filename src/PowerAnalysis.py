'''
This module analyses a gaussian information engine from the module InformationEngine
at different sampling frequencies and calculates, compares the output power of the simulation
to analytically calculated power from the AnalyticPower module.

This module requires the modules matplotlib, numpy, AnalyticPower and InformationEngine.

Classes
-------
Create_Data
    runs the information engine simulation and collects its data
Statistics
    creates statistics on data
Analysor
    analyses data of information engine
Visualisation
    visualises results
'''


import InformationEngine as info
import AnalyticPower as ap
import numpy as np
from matplotlib import pyplot as plt


#get data from simulation - depends on information engine 
class Create_Data:
    '''
    a class used to represent data of a simulated information engine
    
    ...
    
    Attributes
    ----------
    frequency : numpy array
        array of sampling frequencies of ratcheting events (positional update of bead and trap) occur
    length : int
        integer that resembles maximal number of ratcheting events at a single simulation
    
    Methods
    -------
    single_run()
        runs one simulation and collects final bead position
    statistics(amount : int)
        runs multiple simulations and collects final bead position
    '''


    def __init__(self, frequency: np.array, length: int):
        '''initialises  simulation'''

        self.frequency = frequency
        self.length = length
    
        #starts simulation
        self.simulation = info.Simulation(length)  

        #output
        self.positions = []
        self.position = []


    def single_run(self): 
        '''collects data of single simulation run'''     

        #creating raw data
        self.simulation.multiple_trajectories(self.frequency)

        self.position = self.simulation.final_positions
        

    def statistics(self, amount: int):
        '''take statistics of multiple runs'''

        #input
        self.amount = amount
        
        for n in range(amount):

            self.simulation.multiple_trajectories(self.frequency)
            self.positions.append(self.simulation.final_positions)



# do statistics on multiple runs - get data of multiple simulations
class Statistics:
    '''
    a class that represents objects to create statistics of multiple simulation runs regarding different frequencies
    
    ...
    
    Attributes
    ----------
    data : CreateData
        data on which statistics should be made
        
    Methods
    -------
    mean()
        calculates mean of positional data
    std()
        calculates standard deviation of positional data
    ste()
        calculates standard error of mean of positional data
    '''

    def __init__(self, data: Create_Data):
        '''initialises data to create statistics on'''

        #get_data
        self.positions = data.positions
        self.frequency = data.frequency
        self.length = data.simulation.length 

        
    def mean(self):
        '''calculates mean of multiple simulations'''

        return np.mean(self.positions, axis=0)

    def std(self):
        '''calculates standard deviation of multiple simualtions''' 

        return np.std(self.positions, axis=0)

    def ste(self):
        '''calculates standard error'''

        return np.std(self.positions, axis=0, ddof=1)/np.sqrt(np.size(self.positions, axis=0)) 



#analyses results from simulation 
class Analysor:
    '''
    a class that represents objects to analyse a gaussian information engine at different frequencies
    
    ...

    Attributes
    ----------
    data : numpy array
        array with data to be analysed (i.e. bead position)
    frequency : numpy array
        sampling frequencies of ratcheting events
    length : int
        that resembles maximal number of ratcheting events at a single simulation
    
    Methods
    -------
    calc_velocity()
        calculates final velocity of bead depending on frequency
    calc_power()
        calculates final power of bead depending on frequency
    '''

    def __init__(self, data: np.array, frequency: np.array, length: int):
        '''initialises analysis, creates raw data to be analysed'''

        #input
        self.frequencies = frequency
        self.length = length
        self.positions = data


        #output
        self.velocity = np.zeros_like(self.positions)
        self.power = np.zeros_like(self.positions)


    def calc_velocity(self):
        '''calculates velocity'''

        for index, position in enumerate(self.positions):

            self.velocity[index] = position*(self.frequencies[index]/self.length)


    def calc_power(self):
        '''calculates power'''

        self.bead = info.Bead(0, self.length)

        self.calc_velocity()

        for index, velocity in enumerate(self.velocity):

            self.power[index] = velocity*self.bead.delta_g            



#visualise your results from multiple simulations
class Visualisation:
    '''
    a class to represent visualisations of trap simulations
    
    ...

    Attributes
    ----------
    time_length : int
        length of simulation, i.e. number of ratcheting events 
    amount_of_runs : int
        amount of simulation runs to create statistics on   
    frequency : numpy array
        sampling frequency of rachting events
    
    Methods
    -------
    run_analytic_curve() 
        runs analytic calculation to calculate power and calculates theoretical limits
    run_simulation()
        runs simulation and calculates power
    plot_simulation()
        plots simulation
    plot_analytic_curve()
        plots analytic curve
    compare()
        plots both simulation and analytic curve in one plot          

    ''' 

    def __init__(self, time_length: int, amount_of_runs: int, frequency: np.array):
        '''initialises visualisation'''

        #simulation
        self.frequency = frequency
        self.data = Create_Data(frequency, time_length)
        self.amount = amount_of_runs

        
    def run_analytic_curve(self):
        '''runs analytic calculation to calculate power and calculates theoretical limits'''

        analytic_power, frequency = ap.power()

        lim_high = ap.power_limit_high()
        lim_low = ap.power_limit_low(self.frequency)

        return analytic_power, frequency, lim_high, lim_low
    

    def run_simulation(self):    
        '''runs simulation and calculates power'''

        self.data.statistics(self.amount)

        stats = Statistics(self.data)

        stats_mean = stats.mean()
        stats_error = stats.ste()

        #mean of runs
        analysis = Analysor(stats_mean, stats.frequency, stats.length)
        analysis.calc_power()

        #error of runs
        analysis_error = Analysor(stats_error, stats.frequency, stats.length)
        analysis_error.calc_power()

        return analysis.power, analysis_error.power


    def plot_simulation(self):
        '''plots simulation data'''

        power, error = self.run_simulation()

        #simulation
        plt.errorbar(self.frequency, power, yerr=error, fmt='.')

        #plot specifics
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('Power')
        plt.xlabel('Frequency')
        plt.ylim(5e-3,0.5)
        plt.xlim(3e-2,1e3)
        plt.show()


    def plot_analytic_curve(self):
        '''plots analytic curve and theoretical limits'''

        analytic_power, frequency, lim_high, lim_low = self.run_analytic_curve()

        #plot
        plt.plot(frequency, analytic_power)

        #limits
        plt.axline([0,lim_high],slope=0, color='grey', ls='--')
        plt.plot(self.frequency, lim_low, color='grey', ls='--')

        #plot specifics
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('Power')
        plt.xlabel('Frequency')
        plt.ylim(5e-3,0.5)
        plt.xlim(3e-2,1e3)
        plt.show()


    def compare(self):
        '''compare analytic curve with simulated data'''

        #simulated data
        power, error = self.run_simulation()

        #analytical data
        analytic_power, frequency, lim_high, lim_low = self.run_analytic_curve()

        #plot
        plt.errorbar(self.frequency, power, yerr=error, fmt='.', color='darkorange')
        plt.plot(frequency, analytic_power)
        plt.axline([0,lim_high],slope=0, color='grey', ls='--')
        plt.plot(self.frequency, lim_low, color='grey', ls='--')

        #plot specifics
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('Power')
        plt.xlabel('Frequency')
        plt.ylim(5e-3,0.5)
        plt.xlim(3e-2,1e3)
        plt.show()
    


