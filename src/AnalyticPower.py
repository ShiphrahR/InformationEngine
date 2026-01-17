import numpy as np
from scipy.special import erf
from scipy.sparse.linalg import eigs 
from scipy import stats as st


def position_prop(x, u, dt, delta_g):
    mean1 = (u[np.newaxis, :] + delta_g)*np.exp(-dt) - delta_g
    mean2 = - delta_g - (u[np.newaxis, :] - delta_g)*np.exp(-dt)
    variance = np.sqrt(1 - np.exp(-2*dt))
    new_position = (np.heaviside(-u[np.newaxis, :],0)*st.norm.pdf(x[:,np.newaxis], loc=mean1, scale=variance) + np.heaviside(u[np.newaxis, :],0)*st.norm.pdf(x[:,np.newaxis], loc=mean2, scale=variance))*np.abs(x[0] - x[1])
    return new_position

def trap_prop(x, v, dt, delta_g): 
    mean = delta_g - (v[np.newaxis, :] + delta_g)*np.exp(-dt)
    variance = np.sqrt(1 - np.exp(-2*dt))
    new_trap = np.heaviside(-x[:,np.newaxis], 0)*(st.norm.pdf(x[:,np.newaxis], loc=mean, scale=variance) + st.norm.pdf(x[:,np.newaxis], loc=-mean, scale=variance))*np.abs(x[0] - x[1])
    return new_trap

def steady_states(position, dt, delta_g):

    new_position = position_prop(position, position, dt, delta_g)
    new_trap = trap_prop(position, position, dt, delta_g)

    steady_state_particle = eigs(new_position, k=1, sigma=1)[1].reshape(1,-1)
    steady_state_trap = eigs(new_trap, k=1, sigma=1)[1].reshape(1,-1)

    steady_state_particle /= np.trapz(steady_state_particle, position)
    steady_state_trap /= np.trapz(steady_state_trap, position)

    return steady_state_particle.real, steady_state_trap.real


def power(time_steps=30, delta_g=0.84):

    frequency = np.logspace(-3, 3, time_steps)
    total_power = np.zeros(time_steps)

    grid = np.linspace(-20,20,2000)

    for index, freq in enumerate(frequency):

        particle, trap = steady_states(grid, 1/freq, delta_g)

        total_power[index] = delta_g*(np.trapz(particle*grid,grid) - np.trapz(trap*grid,grid))*freq

    return total_power, frequency

def power_limit_high(delta_g=0.8):
    p = np.sqrt(2.0/np.pi)*delta_g*np.exp(-0.5*(delta_g**2))/(1+erf(np.sqrt(0.5)*delta_g))
    return p    

def power_limit_low(frequency, delta_g=0.8):
    w_eq = np.sqrt(2/np.pi)*delta_g*np.exp(-(delta_g**2)*0.5) + (delta_g**2)*(erf(delta_g*np.sqrt(0.5)) - 1)
    p =  frequency*w_eq 
    return p
