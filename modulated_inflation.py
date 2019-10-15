
# coding: utf-8

# In[2]:


import numpy as np
import math 
import time
from scipy import interpolate
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')

A = 0.130364
f = 0.129551
V0 = 2.1*10**-10.0
N_range = np.array([0., 69.]) #Range of values of efolds

V = lambda phi : V0*(np.tanh(phi/math.sqrt(6)) + A*np.sin(np.tanh(phi/math.sqrt(6))/f))**2.
dV = lambda phi : 2*V0*((np.tanh(phi/math.sqrt(6)) + A*np.sin(np.tanh(phi/math.sqrt(6))/f)))*((1/math.sqrt(6))*np.cosh(phi/math.sqrt(6))**-2.
                  + (A/math.sqrt(6*f**2.0))*np.cos(np.tanh(phi/math.sqrt(6))/f)*np.cosh(phi/math.sqrt(6))**-2.0)

phi0 = 6.2
Dphi0 = -dV(phi0)/V(phi0)

#Define the second derivative
def DDphi(N, phi, Dphi):
	return -3*Dphi + 0.5*(Dphi)**3. - (3 - 0.5*Dphi**2.)*(dV(phi)/V(phi))

def RK4_inflation(N_range, phi0_, Dphi0_, DDphi_,step):
	n = int((N_range[-1] - N_range[0])/step)
	N_in = N_range[0]
	phi_in = phi0_ 
	Dphi_in = Dphi0_ 

	N_sol = np.empty(0)
	N_sol = np.append(N_sol, N_in)

	phi_sol = np.empty(0)
	phi_sol = np.append(phi_sol, phi_in)

	Dphi_sol = np.empty(0)
	Dphi_sol = np.append(Dphi_sol, Dphi_in)

	for i in range(n):
		k1 = Dphi_sol[i]
		K1 = DDphi_(N_sol[i], phi_sol[i], Dphi_sol[i])
		k2 = Dphi_sol[i] + 0.5*step*K1 
		K2 = DDphi_(N_sol[i] + 0.5*step, phi_sol[i] + 0.5*step*k1,  Dphi_sol[i] + 0.5*step*K1)
		k3 = Dphi_sol[i] + 0.5*step*K2
		K3 = DDphi_(N_sol[i] + 0.5*step, phi_sol[i] + 0.5*step*k2,  Dphi_sol[i] + 0.5*step*K2)
		k4 = Dphi_sol[i] + step*K3 
		K4 = DDphi_(N_sol[i] + step, phi_sol[i] + step*k3, Dphi_sol[i] + step*K3)

		phi_temp = phi_sol[i] + (step/6.)*(k1 + 2*k2 + 2*k3 + k4)
		Dphi_temp = Dphi_sol[i] + (step/6.)*(K1 + 2*K2 + 2*K3 + K4)
		phi_sol = np.append(phi_sol, phi_temp)
		Dphi_sol = np.append(Dphi_sol, Dphi_temp)

		N_temp = N_sol[i] + step 
		N_sol = np.append(N_sol, N_temp)
	return [N_sol, phi_sol, Dphi_sol]

h = 0.05
[N, phi, Dphi] = RK4_inflation(N_range, phi0, Dphi0, DDphi, h)
        
#Create interpolating functions such that quantities can be treated as functions of efoldings
DDDphi = np.gradient(DDphi(N, phi, Dphi), 0.05)
PHI = interpolate.interp1d(N, phi, kind = 'cubic')    
DPHI = interpolate.interp1d(N, Dphi, kind = 'cubic')
DDPHI = interpolate.interp1d(N, DDphi(N, phi, Dphi), kind = 'cubic')
DDDPHI = interpolate.interp1d(N, DDDphi, kind = 'cubic')

eps = lambda N: 0.5*DPHI(N)**2.
eta = lambda N: eps(N) - (DPHI(N)*DDPHI(N))/(2*eps(N))
H = lambda N: np.sqrt(V(PHI(N))/(3 - eps(N)))

a0 = 0.00738651
a = lambda N: a0*np.exp(N)


# In[3]:


plt.plot(N, PHI(N), label=r'$\phi$', color='b')
plt.plot(N, DPHI(N), label=r'$\frac{d\phi}{dN}$', color='r')
plt.xlabel(r'$N$')
plt.legend()
plt.title('Inflaton field evolution')
plt.show()


# In[4]:


plt.plot(N, eps(N), label=r'$\epsilon$', color='b')
plt.plot(N, np.absolute(eta(N)), label=r'$|\eta|$', color='r')
plt.yscale('log')
plt.legend()
plt.title('Hubble flow parameters')
plt.show()


# In[5]:


d_eps = np.gradient(0.5*Dphi**2., 0.05)
D_eps = interpolate.interp1d(N, d_eps, kind = 'cubic')
d_eta = np.gradient(0.5*Dphi**2. - (DDphi(N, phi, Dphi)/Dphi), 0.05)
D_eta = interpolate.interp1d(N, d_eta, kind = 'cubic')

t_start = time.time()

def DDuk(k, N, uk, Duk):
    '''
    returns the value of the second derivative 
    of the mode functions
    '''
    return -(1 - eps(N))*Duk - ((k**2.)/(a(N)*H(N))**2.)*uk - (1 + eps(N) - eta(N))*(eta(N) - 2)*uk + D_eps(N)*uk - D_eta(N)*uk

'''
Solves for N when k = 100aH and k = 0.01aH
Computations extremely sensitive to step size
'''
def solve_Ninit(k, N_array):
	Ni = N_array[0]
	step = N_array[1] - N_array[0]
	Ninit_temp = np.asarray([k - 100.*a(N)*H(N) for N in N_array])
	Ninit_test = np.where(Ninit_temp > 0)
	return Ni + Ninit_test[0][-1]*step

def solve_Nfin(k, N_array):
	Ni = N_array[0]
	step = N_array[1] - N_array[0]
	Nfin_temp = np.asarray([k - 0.01*a(N)*H(N) for N in N_array])
	Nfin_test = np.where(Nfin_temp > 0)
	return Ni + Nfin_test[0][-1]*step

'''
Sets up Bunch-Davies initial conditions
to uk and duk/dN
'''
def uk_init(k):
	uk0 = np.zeros(1, dtype = complex)
	uk0.real = (2*k)**-0.5
	return uk0

def duk_init(k):
	duk0 = np.zeros(1, dtype = complex)
	duk0.imag = -(k**0.5)/(math.sqrt(2)*0.01*k)
	return duk0

def rk4_Mukhanov_Sasaki(Ninit_, Nfin_, k, uk0, duk0, DDuk_, step):
	n = int((Nfin_ - Ninit_)/step)
	N_in = Ninit_
	uk_in = uk0 
	Duk_in = duk0 

	N_sol = np.empty(0)
	N_sol = np.append(N_sol, N_in)
	uk_sol = np.empty(0, dtype = complex)
	uk_sol = np.append(uk_sol, uk_in)
	Duk_sol = np.empty(0, dtype = complex)
	Duk_sol = np.append(Duk_sol, Duk_in)

	for i in range(n):
		k1 = Duk_sol[i]
		K1 = DDuk_(k, N_sol[i], uk_sol[i], Duk_sol[i])
		k2 = Duk_sol[i] + 0.5*step*K1
		K2 = DDuk_(k, N_sol[i] + 0.5*step, uk_sol[i] + 0.5*step*k1, Duk_sol[i] + 0.5*step*K1)
		k3 = Duk_sol[i] + 0.5*step*K2
		K3 = DDuk_(k, N_sol[i] + 0.5*step, uk_sol[i] + 0.5*step*k2, Duk_sol[i] + 0.5*step*K2)
		k4 = Duk_sol[i] + step*K3
		K4 = DDuk_(k, N_sol[i] + step, uk_sol[i] + step*k3, Duk_sol[i] + step*K3)

		uk_temp = uk_sol[i] + (step/6.)*(k1 + 2*k2 + 2*k3 + k4)
		Duk_temp = Duk_sol[i] + (step/6.)*(K1 + 2*K2 + 2*K3 + K4)
		uk_sol = np.append(uk_sol, uk_temp)
		Duk_sol = np.append(Duk_sol, Duk_temp)
		N_temp = N_sol[i] + step 
		N_sol = np.append(N_sol, N_temp)
	return [N_sol, uk_sol, Duk_sol]

k = 10**20
step = 0.005 #this step produces an accurate calculation of the power spectrum
N_init = solve_Ninit(k, N)
N_fin = solve_Nfin(k, N)
uk0 = uk_init(k)
duk0 = duk_init(k)

[N_sol, uk_sol, Duk_sol] = rk4_Mukhanov_Sasaki(N_init, N_fin, k, uk0, duk0, DDuk, step)
Pspec = ((k**3.)/(2*math.pi**2))*(np.absolute(uk_sol)/(a(N_sol)*DPHI(N_sol)))**2.
plt.plot(N_sol, Pspec)
plt.yscale('log')
plt.show()

print("The value of the curvature perturbation is " + str(Pspec[-1]))

t_end = time.time()
print("Duration of execution: " + str(t_end - t_start) + " seconds")


# In[12]:


'''
Define a logarithmically spaced list of k-values
to be looped for in the computation of the
curvature power spectrum
'''
ti = time.time()
k_list = 5*np.logspace(-2, 21, 500)

def Power_Spectrum(k_list, N):
    PSpec = np.empty(0)
    for i in range(len(k_list)):
        step_ = 0.005
        k_ = k_list[i]
        Ni_ = solve_Ninit(k_, N)
        Nf_ = solve_Nfin(k_, N)
        uk0_ = uk_init(k_)
        duk0_ = duk_init(k_)
        [N_, uk_, Duk_] = rk4_Mukhanov_Sasaki(Ni_, Nf_, k_, uk0_, duk0_, DDuk, step_)
        PSpec_temp = ((k_**3.)/(2*math.pi**2.))*(np.absolute(uk_)/(a(N_)*DPHI(N_)))**2.
        PSpec = np.append(PSpec, PSpec_temp[-1])
        print("Loop no. " + str(i) + " completed")
    print("All loops have been completed")
    return PSpec

Pow_Spec = Power_Spectrum(k_list, N)
tf = time.time()
print("Time for completion " + str(tf - ti) + " seconds")


# In[22]:


plt.scatter(k_list, Pow_Spec, s = 5, c = 'b')
plt.xscale('log')
plt.yscale('log')
plt.ylim(10**-12, 10**0)
plt.xlabel(r'$k$', fontsize = 14)
plt.ylabel(r'$\mathcal{P}_{\mathcal{R}}$', fontsize = 14)
plt.show()

