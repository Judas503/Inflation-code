
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import sympy as sp
import math
import time
from scipy import interpolate
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')

V0 = 2.1*10**-10.
c0 = 0.16401
c1 = 0.3
c2 = -1.426
c3 = 2.20313

'''
Symbolically define V(phi) sympy to be able to calculate the derivative; 
then they are turned into lambda functions
'''
phi = sp.symbols('phi')
V_sym = V0*(c0 + c1*sp.tanh(phi/np.sqrt(6)) + c2*(sp.tanh(phi/np.sqrt(6)))**2. + c3*(sp.tanh(phi/np.sqrt(6)))**3.)**2.
V = sp.lambdify(phi, V_sym, "numpy")

dV_sym = sp.diff(V_sym, phi)
dV = sp.lambdify(phi, dV_sym, "numpy")

N_range = np.array([0., 65.8])
phi0 = 7.35
dphi0 = -dV(phi0)/V(phi0)

def ddphi(N_, phi_, dphi_):
    return -3*dphi_ + 0.5*(dphi_)**3. - (3 - 0.5*(dphi_)**2.)*(dV(phi_)/V(phi_))

def RK4_inflation(N_range, phi0_, dphi0_, ddphi_, step_):
    n = int((N_range[-1] - N_range[0])/step_)
    
    N_in = N_range[0]
    phi_in = phi0_
    dphi_in = dphi0_
    
    N_sol = np.empty(0)
    N_sol = np.append(N_sol, N_in)
    
    phi_sol = np.empty(0, dtype = float)
    phi_sol = np.append(phi_sol, phi_in)
    
    dphi_sol = np.empty(0, dtype = float)
    dphi_sol = np.append(dphi_sol, dphi_in)
    
    for i in range(n):
        k1 = dphi_sol[i]
        K1 = ddphi_(N_sol[i], phi_sol[i], dphi_sol[i])
        k2 = dphi_sol[i] + 0.5*step_*K1
        K2 = ddphi_(N_sol[i] + 0.5*step_, phi_sol[i] + 0.5*step_*k1, dphi_sol[i] + 0.5*step_*K1)
        k3 = dphi_sol[i] + 0.5*step_*K2
        K3 = ddphi_(N_sol[i] + 0.5*step_, phi_sol[i] + 0.5*step_*k2, dphi_sol[i] + 0.5*step_*K2)
        k4 = dphi_sol[i] + step_*K3
        K4 = ddphi_(N_sol[i] + step_, phi_sol[i] + step_*k3, dphi_sol[i] + step_*K3)
        
        phi_temp = phi_sol[i] + (step_/6.)*(k1 + 2*k2 + 2*k3 + k4)
        dphi_temp = dphi_sol[i] + (step_/6.)*(K1 + 2*K2 + 2*K3 + K4)
        phi_sol = np.append(phi_sol, phi_temp)
        dphi_sol = np.append(dphi_sol, dphi_temp)
        
        N_temp = N_sol[i] + step_
        N_sol = np.append(N_sol, N_temp)
    return [N_sol, phi_sol, dphi_sol]

step = 0.05
[N, phi, dphi] = RK4_inflation(N_range, phi0, dphi0, ddphi, step)

plt.plot(N, phi, label = r'$\phi$', color = 'b')
plt.plot(N, dphi, label = r'$\frac{d\phi}{dN}$', color = 'r')
plt.xlabel(r'$N$', fontsize = 16)
plt.legend()
plt.title('Inflaton field evolution')
plt.show()


# In[16]:


'''
Create interpolating functions phi and dphi so that
they can be expressed as functions of efolding
'''
Phi = interpolate.interp1d(N, phi, kind = 'cubic', fill_value = 'extrapolate')
DPhi = interpolate.interp1d(N, dphi, kind = 'cubic', fill_value = 'extrapolate')
DDPhi = interpolate.interp1d(N, ddphi(N, phi, dphi), kind = 'cubic', fill_value = 'extrapolate')
ddphi_deriv = np.gradient(ddphi(N, phi, dphi), 0.05)
DDDPhi = interpolate.interp1d(N, ddphi_deriv, kind = 'cubic', fill_value = 'extrapolate')

eps = lambda N: 0.5*DPhi(N)**2.
eta = lambda N: eps(N) - (DPhi(N)*DDPhi(N))/(2*eps(N))
H = lambda N: np.sqrt(V(Phi(N))/(3 - eps(N)))

plt.plot(N, eps(N), label = r'$\epsilon$', color = 'b')
plt.plot(N, np.absolute(eta(N)), label = r'$|\eta|$', color = 'r')
plt.xlabel(r'$N$')
plt.yscale('log')
plt.legend()
plt.title('Hubble flow parameters')
plt.show()


# In[18]:


d_eps = np.gradient(0.5*dphi**2., 0.05)
d_eta = np.gradient(0.5*dphi**2 - (ddphi(N, phi, dphi)/dphi), 0.05)
D_eps = interpolate.interp1d(N, d_eps, kind = 'cubic', fill_value = 'extrapolate')
D_eta = interpolate.interp1d(N, d_eta, kind = 'cubic', fill_value = 'extrapolate')

a = lambda N: 0.0450899*np.exp(N)

t_start = time.time()
def DDuk(k_, N_, uk_, Duk_):
    return -(1 - eps(N_))*Duk_ - ((k_**2.)/(a(N_)*H(N_))**2.)*uk_ - (1 + eps(N_) - eta(N_))*(eta(N_) - 2)*uk_ + D_eps(N_)*uk_- D_eta(N_)*uk_

def solve_Ninit(k_, N_array):
    Ni = N_array[0]
    step = N_array[1] - N_array[0]
    Ninit_temp = np.asarray([k_ - 100.*a(N)*H(N) for N in N_array])
    Ninit_test = np.where(Ninit_temp > 0)
    return Ni + Ninit_test[0][-1]*step

def solve_Nfin(k_, N_array):
    Ni = N_array[0]
    step = N_array[1] - N_array[0]
    Nfin_temp = np.asarray([k_ - 0.01*a(N)*H(N) for N in N_array])
    Nfin_test = np.where(Nfin_temp > 0)
    return Ni + Nfin_test[0][-1]*step

def uk_init(k_):
    uk0 = np.zeros(1, dtype = complex)
    uk0.real = (2*k_)**-0.5
    return uk0

def Duk_init(k_):
    Duk0 = np.zeros(1, dtype = complex)
    Duk0.imag = -(k_**0.5)/(np.sqrt(2)*0.01*k_)
    return Duk0

def RK4_Mukhanov_Sasaki(Ninit_, Nfin_, k_, uk0_, Duk0_, DDuk_, step_):
    n = int((Nfin_ - Ninit_)/step_)
    N_in = Ninit_
    uk_in = uk0_
    Duk_in = Duk0_
    
    N_sol = np.empty(0)
    N_sol = np.append(N_sol, N_in)
    
    uk_sol = np.empty(0, dtype = complex)
    uk_sol = np.append(uk_sol, uk_in)
    
    Duk_sol = np.empty(0, dtype = complex)
    Duk_sol = np.append(Duk_sol, Duk_in)
    
    for i in range(n):
        k1 = Duk_sol[i]
        K1 = DDuk_(k_, N_sol[i], uk_sol[i], Duk_sol[i])
        k2 = Duk_sol[i] + 0.5*step_*K1
        K2 = DDuk_(k_, N_sol[i] + 0.5*step_, uk_sol[i] + 0.5*step_*k1, Duk_sol[i] + 0.5*step_*K1)
        k3 = Duk_sol[i] + 0.5*step_*K2
        K3 = DDuk_(k_, N_sol[i] + 0.5*step_, uk_sol[i] + 0.5*step_*k2, Duk_sol[i] + 0.5*step_*K2)
        k4 = Duk_sol[i] + step_*K3
        K4 = DDuk_(k_, N_sol[i] + step_, uk_sol[i] + step_*k3, Duk_sol[i] + step_*K3)
        
        N_temp = N_sol[i] + step_
        N_sol = np.append(N_sol, N_temp)
        
        uk_temp = uk_sol[i] + (step_/6.)*(k1 + 2*k2 + 2*k3 + k4)
        uk_sol = np.append(uk_sol, uk_temp)
        
        Duk_temp = Duk_sol[i] + (step_/6.)*(K1 + 2*K2 + 2*K3 + K4)
        Duk_sol = np.append(Duk_sol, Duk_temp)
    return [N_sol, uk_sol, Duk_sol]

k = 5*10**20.
step = 0.005
N_init = solve_Ninit(k, N)
N_fin = solve_Nfin(k, N)
uk0 = uk_init(k)
Duk0 = Duk_init(k)

[N_sol, uk_sol, Duk_sol] = RK4_Mukhanov_Sasaki(N_init, N_fin, k, uk0, Duk0, DDuk, step)
Pspec = ((k**3.)/(2*np.pi**2))*(np.absolute(uk_sol)/(a(N_sol)*DPhi(N_sol)))**2.

plt.plot(N_sol, Pspec)
plt.yscale('log')
plt.show()

print("The value of the curvature perturbation is " + str(Pspec[-1]))

t_end = time.time()
print("Duration of execution: " + str(t_end - t_start) + " seconds")


# In[19]:


ti = time.time()
k_list = 5*np.logspace(-2, 20, 500)

def Power_Spectrum(k_list, N):
    PSpec = np.empty(0)
    for i in range(len(k_list)):
        step_ = 0.005
        k_ = k_list[i]
        Ni_ = solve_Ninit(k_, N)
        Nf_ = solve_Nfin(k_, N)
        uk0_ = uk_init(k_)
        duk0_ = Duk_init(k_)
        [N_, uk_, Duk_] = RK4_Mukhanov_Sasaki(Ni_, Nf_, k_, uk0_, duk0_, DDuk, step_)
        PSpec_temp = ((k_**3.)/(2*np.pi**2.))*(np.absolute(uk_)/(a(N_)*DPhi(N_)))**2.
        PSpec = np.append(PSpec, PSpec_temp[-1])
        print("Loop no. " + str(i + 1) + " completed")
    print("All loops have been completed")
    return PSpec

Pow_Spec = Power_Spectrum(k_list, N)
tf = time.time()
print("Time for completion " + str(tf - ti) + " seconds")


# In[24]:


plt.scatter(k_list, Pow_Spec, s = 5, c = 'b')
plt.xscale('log')
plt.yscale('log')
plt.ylim(10**-14, 10**0)
plt.xlabel(r'$k$', fontsize = 14)
plt.ylabel(r'$\mathcal{P}_{\mathcal{R}}$', fontsize = 14)
plt.title('Curvature Power Spectrum', fontsize = 16)
plt.show()

