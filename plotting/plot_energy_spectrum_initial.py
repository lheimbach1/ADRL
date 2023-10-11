import numpy as np
import matplotlib.pyplot as plt
import os

def compute_D(u,k,dk,nu):
    u_hat = np.fft.fft(u)
    u_hat/=u.shape[0]
    return nu*np.sum(k**2*np.abs(u_hat)) * dk

def compute_energy_spectrum(array):
    nx = array.shape[0]
    array_hat = np.fft.fft(array)
    array_hat /= nx
    # Energy Spectrum
    espec = 0.5 * np.absolute(array_hat)**2 # equation (72)
    # Angle Averaging
    eplot = np.zeros(nx // 2, dtype='double')
    for i in range(1, nx // 2):
        eplot[i] = 0.5 * (espec[i] + espec[nx - i]) # average positive and negative energy frequencies
    return eplot

def compute_E(k):
    k_0 = 10
    A = (2 * k_0**(-5)) / (3 * np.sqrt(np.pi))
    E_k = A * k**4 * np.exp(-(k/k_0)**2)
    return E_k

def initalize_paper(N,random_dict):
    dx = 2*np.pi/N
    k = np.fft.fftfreq(N,dx)*2*np.pi
    dk = k[1] - k[0]
    E_k = compute_E(k)
    random_arr = np.array([random_dict[np.abs(val)] for val in k])
    u_hat = np.zeros(N,dtype = complex)
    u_hat[:N//2] = np.sqrt(2*E_k[:N//2])*(np.cos(2*np.pi*random_arr[:N//2]) + 1j*np.sin(2*np.pi*random_arr[:N//2]))
    u_hat[N//2:] = np.sqrt(2*E_k[N//2:])*(np.cos(2*np.pi*random_arr[N//2:]) - 1j*np.sin(2*np.pi*random_arr[N//2:]))
    u_hat*=N
    u = np.fft.ifft(u_hat)
    
    return np.real(u), dx, k, dk

N_LES = 32768
N_DNS = int(N_LES*4)
label_fontsize = 16
legend_fontsize = 12

dx = 2*np.pi/N_DNS
k = np.fft.fftfreq(N_DNS,dx)*2*np.pi
random_dict = {abs_val: np.random.uniform(0, 1) for abs_val in np.abs(k)}
u_DNS, dx_DNS, k_DNS, dk = initalize_paper(N_DNS,random_dict)
u_LES, dx_LES, k_LES, dk = initalize_paper(N_LES,random_dict)

dir_path = f'../figures/report_plots'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

e_LES = compute_energy_spectrum(u_LES)
e_DNS = compute_energy_spectrum(u_DNS)
plt.loglog(k_DNS[:N_DNS // 2],e_DNS)
plt.legend()
plt.xlabel("k", fontsize=label_fontsize)
plt.ylabel("E(k)", fontsize=label_fontsize)
plt.xticks(fontsize=label_fontsize)
plt.yticks(fontsize=label_fontsize)
plt.tight_layout()
plt.savefig(f'{dir_path}/inital_e_spectrum.png', dpi=600)  
plt.show()          
plt.close()

