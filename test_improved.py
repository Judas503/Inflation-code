'''
Stochastic inflation code using a quadratic potential without using the slow-roll
approximation for the Hubble parameter and using the constant mass solution of
the curvature power spectrum in the definition of the noise term
'''
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.special import gamma
from numba import jit
import time

m = 0.01
N_test = 60.0 
dN = 0.01
n = int(N_test/dN)
N = np.linspace(0.0, N_test, n)

'''
Defining the asymptotic form of the Hankel functions for the constant
mass solutions of the curvature power spectrum
'''
hankelFunc = lambda x: (gamma(2.5)**-2.)*(0.5*x)**3. + ((gamma(1.5)/np.pi)**2.)*(2/x)**3.

@jit(nopython=True)
def efold_return(_sigma, _phi_in, _dphi_in):
	efold_count = 0
	phi = np.zeros(n)
	phi_deriv = np.zeros(n)
	
	H = np.zeros(n)
	phi[0] = _phi_in 
	phi_deriv[0] = _dphi_in
	H[0] = np.sqrt((phi_deriv[0]**2.)/6. + ((m**2.)*phi[0]**2.)/6.) 

	for i in range(n-1):
		phi[i+1] = phi[i] + (phi_deriv[i]/H[i])*dN + np.sqrt(((H[i]**2.)/(8*np.pi))*(_sigma**3)*hankelFunc(_sigma))*np.sqrt(dN)*np.random.randn()
		
		phi_deriv[i+1] = phi_deriv[i] - 3*phi_deriv[i]*dN - ((phi[i]*m**2.)/H[i])*dN
		
		H[i+1] = np.sqrt(phi_deriv[i+1]**2. + (m*phi[i+1])**2.)/np.sqrt(6.)

		if (phi[i+1] - phi[i])**2. >= 2*dN**2.:
			break 
		else:
			efold_count += efold_count

	return efold_count 

def stat_return(num1, num2):
	N_store = np.zeros((num1, num2))
    phi_in_list = np.linspace(1, 15, num1)

    for i in range(0, num1):
        print(i)
        for j in range(num2):
            N_store[i, j] = efold_return(phi_in_list[i], 0)
    return N_store
