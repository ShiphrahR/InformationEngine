import sys
sys.path.append("/home/shiphrah/Desktop/Univers/Orientierungspraktikum/Projekt: Info_Engines/Projekt.git/src")

import FeedbackAnalysis as fa
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize as opt

def apparent_work(alpha: float, sigma: float):
    '''calculates apparent work depending on feedback-gain alpha'''

    analyse = fa.Analysor(sigma, alpha, 0, 0, 1)
    work = analyse.trap_work_rate()

    return work


def search_alpha(sigma: float):
    '''finds alpha such that the apparent work is close to zero depending on sigma'''
    
    if sigma > 1.4:
        alpha = 0

    if sigma < 0.001:
        alpha = 2
        
    else: 
        alpha = opt.newton(apparent_work, x0=0.01, x1=1.8, tol=1e-3, maxiter=500, args=(sigma,))

    return alpha



def search_alphas(sigmas: np.array):
    '''finds alphas to an array of sigmas such that the apparent work is zero'''

    alphas = np.zeros_like(sigmas)

    for idx, itm in enumerate(sigmas):

        alphas[idx] = search_alpha(itm)

    return alphas    


sigmas = np.logspace(-2, 0.3, 10)
search_alphas(sigmas)