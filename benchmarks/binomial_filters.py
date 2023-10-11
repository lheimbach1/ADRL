import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags, csc_matrix
import os
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
from scipy.special import comb

#Calculates Energy Dissipation
def compute_D(u,k,dk,nu):
    u_hat = np.fft.fft(u)
    u_hat/=u.shape[0]
    return nu*np.sum(k**2*np.abs(u_hat)) * dk

#Computes the Energy Spectrum of the Field "array"
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

#Computes the Total Energy 
def compute_total_energy(u,df):
    energy_spectrum = compute_energy_spectrum(u)
    return np.sum(energy_spectrum) * df * 2 # to account for averaging of  positive and negative frequencies

#Computes Initial Energy Spectrum
def compute_E(k):
    k_0 = 10
    A = (2 * k_0**(-5)) / (3 * np.sqrt(np.pi))
    E_k = A * k**4 * np.exp(-(k/k_0)**2)
    return E_k

#Sample Field in Physical Space from Inital Energy Spectrum
def initalize_paper(N,random_dict):
    dx = 2*np.pi/N
    k = np.fft.fftfreq(N,dx)*2*np.pi
    dk = k[1] - k[0]
    E_k = compute_E(k)
    # random_dict = {abs_val: np.random.uniform(0, 1) for abs_val in np.abs(k)}
    random_arr = np.array([random_dict[np.abs(val)] for val in k])
    u_hat = np.zeros(N,dtype = complex)
    u_hat[:N//2] = np.sqrt(2*E_k[:N//2])*(np.cos(2*np.pi*random_arr[:N//2]) + 1j*np.sin(2*np.pi*random_arr[:N//2]))
    u_hat[N//2:] = np.sqrt(2*E_k[N//2:])*(np.cos(2*np.pi*random_arr[N//2:]) - 1j*np.sin(2*np.pi*random_arr[N//2:]))
    u_hat*=N
    u = np.fft.ifft(u_hat)
    
    return np.real(u), dx, k, dk

#Computes First Derivative With Sixth-Order Compact Scheme
def sixth_order_scheme_first_derivative(f, h):
    n = len(f)
    rhs = np.zeros_like(f)
    rhs = 14/9 * (np.roll(f, -1) - np.roll(f, 1)) / (2*h) + 1/9 * (np.roll(f, -2) - np.roll(f, 2)) / (4*h)
    diagonals = [np.ones(n), np.ones(n-1)/3, np.ones(n-1)/3, 1/3, 1/3]
    positions = [0, 1, -1, -(n-1), n-1]
    A_sparse = diags(diagonals, positions, shape=(n, n))
    A = csc_matrix(A_sparse)
    f_prime = spsolve(A, rhs)

    return f_prime

#Computes Second Derivative With Sixth-Order Compact Scheme
def sixth_order_scheme_second_derivative(f, h):
    n = len(f)
    rhs = np.zeros_like(f)
    rhs = 12/11 * (np.roll(f, -1) - 2*f + np.roll(f, 1)) / (h**2) + 3/11 * (np.roll(f, -2) - 2*f + np.roll(f, 2)) / (4*h**2)
    diagonals = [np.ones(n), np.ones(n-1)*2/11, np.ones(n-1)*2/11, 2/11, 2/11]
    positions = [0, 1, -1, -(n-1), n-1]
    A_sparse = diags(diagonals, positions, shape=(n, n))
    A = csc_matrix(A_sparse)
    f_double_prime = spsolve(A, rhs)

    return f_double_prime

#Computes Deconvolved Field Theta
def van_cittert(u, G, beta, Q):
    # Initialize theta_0
    theta_prev = u.copy()

    for i in range(Q):
        conv_result = convolve(theta_prev, G)
        update_term = beta * (u - conv_result)
        theta_new = theta_prev + update_term

    return theta_new

#Computes Convolved with Filter "G"
def convolve(u, G):
    return np.convolve(u, G, mode='same')

#Computes One Time-Step with the Third-Order TVD Runge-Kutta Scheme
def tvdrk3_burgers(u, dt, dx, nu):
    #Calculate RHS of Burger's Equation
    def rhs(u):
        u_squared_prime = sixth_order_scheme_first_derivative(u**2, dx)
        u_double_prime = sixth_order_scheme_second_derivative(u, dx)
        return -0.5*u_squared_prime  + (nu)*u_double_prime

    u1 = u.copy()
    u1 += dt * rhs(u)
    u2 = 3/4 * u + 1/4 * u1 + 1/4 * dt * rhs(u1)
    u_new = 1/3 * u + 2/3 * u2 + 2/3 * dt * rhs(u2)

    return u_new

#Computes One Time-Step with the Third-Order TVD Runge-Kutta Scheme with Approximate Deconvolution
def tvdrk3_burgers_explicit_filter(u, dt, dx, nu, beta, Q, G):
    #Calculate RHS of Burger's Equation with Approximate Deconvolution
    def rhs(u):
        theta = van_cittert(u,G,beta,Q)
        u_squared_prime = sixth_order_scheme_first_derivative(theta**2, dx)
        u_double_prime = sixth_order_scheme_second_derivative(u, dx)
        u_squared_prime = convolve(u_squared_prime,G)
        return -0.5*u_squared_prime  + (nu)*u_double_prime

    u1 = u.copy()
    u1 += dt * rhs(u)
    u2 = 3/4 * u + 1/4 * u1 + 1/4 * dt * rhs(u1)
    u_new = 1/3 * u + 2/3 * u2 + 2/3 * dt * rhs(u2)

    return u_new

#Calculates Reward between LES and DNS Energy Spectrum
def reward_function(u_LES, u_DNS, tolerance):
    u_DNS_trimmed = u_DNS[:len(u_LES)]
    
    # Calculate relative errors
    relative_errors = np.zeros(len(u_LES))
    valid_indices = u_DNS_trimmed != 0
    relative_errors[valid_indices] = np.abs(u_LES[valid_indices] - u_DNS_trimmed[valid_indices]) / np.abs(u_DNS_trimmed[valid_indices])
    rewards = np.where(relative_errors < tolerance, 1, -1)
    reward = np.sum(rewards)/rewards.shape[0]

    return reward    

#Returns Array Indicating which Wave-Numbers have Positive and Negative Reward
def reward_array(u_LES, u_DNS, tolerance):
    u_DNS_trimmed = u_DNS[:len(u_LES)]
    # Calculate relative errors
    relative_errors = np.zeros(len(u_LES))
    valid_indices = u_DNS_trimmed != 0
    relative_errors[valid_indices] = np.abs(u_LES[valid_indices] - u_DNS_trimmed[valid_indices]) / np.abs(u_DNS_trimmed[valid_indices])
    
    rewards = np.where(relative_errors < tolerance, 5e-2, 5e-15)
    
    return rewards

#Initializes Odd-Stencil Binomial Filters up to "max_order"
def initialize_binomial_filters(max_order):
    filters = []
    for n in range(0, max_order+1, 2):  # Step by 2 to get only even orders
        m = 1 / (2**n)
        d = [comb(n, k) for k in range(n+1)]
        filters.append(m * np.array(d))

    return filters

#Returns a Odd-Lengthed Array of Zeros with 1 in the Middle Index
def initialize_odd_array(length):
    middle_index = length // 2
    array = [0] * length  
    array[middle_index] = 1

    return np.array(array)

#Initializes Odd-Stencil Binomial Filters up to "(n,l)"
def compute_filters(n, l):
    # Define the binomial filter B^2
    B2 = np.array([0.25, 0.5, 0.25])

    # Define the identity filter
    identity_filter = np.array([0, 1, 0])

    # Initialize a list to store filter arrays
    filter_list = []

    for i in range(2, n+1):        
        # Compute the filter coefficients
        f1 = identity_filter - B2
        coefficients = f1

        for _ in range(i-1):
            coefficients = np.convolve(f1, coefficients, mode='full')

        # Compute the final filter
        f2 = initialize_odd_array(len(coefficients)) - coefficients
        result = f2
        for _ in range(l - 1):
            result = np.convolve(f2, result, mode='full')

        # Normalize the filter
        result /= np.sum(result)

        filter_list.append(result)

    return filter_list

class simulation_env_binomial(gym.Env):
    def __init__(self):
        self.time_step = 0
        self.render_num=0
        self.nu = 5e-4
        self.dt = 1e-4
        self.nt = 10
        self.total_time_steps = 500
        self.N_LES = 512
        self.N_DNS = int(self.N_LES*4)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.N_LES,),dtype=np.float64)
        self.action_space = spaces.Discrete(5)
        dx = 2*np.pi/self.N_DNS
        k = np.fft.fftfreq(self.N_DNS,dx)*2*np.pi
        random_dict = {abs_val: np.random.uniform(0, 1) for abs_val in np.abs(k)}
        binomial_filters = initialize_binomial_filters(64)
        smoothing_filters = compute_filters(33, 1)
        self.combined_filters = binomial_filters + smoothing_filters

        binomial_indices = [0, 1, 2,len(binomial_filters) - 1]

        # Indices for smoothing_filters within self.combined_filters
        # We add len(binomial_filters) to each index to account for the shift.
        smoothing_indices = [len(binomial_filters), 
                            len(binomial_filters) + 1, 
                            len(binomial_filters) + len(smoothing_filters) - 1]

        # Combine the indices
        self.combined_indices = binomial_indices + smoothing_indices
        self.u_DNS, self.dx_DNS, self.k_DNS, dk = initalize_paper(self.N_DNS,random_dict)
        self.u_LES, self.dx_LES, self.k_LES, dk = initalize_paper(self.N_LES,random_dict)
        self.beta = 1.
        self.Q = 2
        self.tolerance = 0.3
        self.cum_reward = 0

    def step(self,action):
        G = self.combined_filters[action]

        for t in range(self.nt):
            self.u_DNS = tvdrk3_burgers(self.u_DNS, self.dt, self.dx_DNS, self.nu)
            self.u_LES = tvdrk3_burgers_explicit_filter(self.u_LES, self.dt, self.dx_LES, self.nu, self.beta, self.Q, G)
            self.time_step += 1
        
        reward = reward_function(compute_energy_spectrum(self.u_LES), compute_energy_spectrum(self.u_DNS),self.tolerance)
        self.cum_reward += reward

        if self.time_step == self.total_time_steps:
            done = True 
            truncated = True
        else: 
            done = False
            truncated = False
        info = {} 
        return compute_energy_spectrum(self.u_LES), compute_energy_spectrum(self.u_DNS), reward, done, truncated, info

    def reset(self,seed=None,options=None):
        np.random.seed(seed)
        self.time_step = 0
        dx = 2*np.pi/self.N_DNS
        k = np.fft.fftfreq(self.N_DNS,dx)*2*np.pi
        random_dict = {abs_val: np.random.uniform(0, 1) for abs_val in np.abs(k)}
        self.u_DNS, self.dx_DNS, self.k_DNS, dk = initalize_paper(self.N_DNS,random_dict)
        self.u_LES, self.dx_LES, self.k_LES, dk = initalize_paper(self.N_LES,random_dict)
        info = {}

        return self.u_LES, info
    
    def render(self, mode='human'):
        if mode == 'human':
            directory_path = f'../figures/benchmarks/binomial_{self.filter_id}_std/'
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            plt.loglog(self.k_DNS[:self.N_DNS // 2],compute_energy_spectrum(self.u_DNS),label = f"N={self.N_DNS}")
            plt.loglog(self.k_LES[:self.N_LES // 2],compute_energy_spectrum(self.u_LES),label = f"N={self.N_LES}")
            plt.xlabel("k")
            plt.ylabel("E(k)")
            plt.title(f"Reward={self.cum_reward}")
            plt.legend()
            plt.savefig(os.path.join(directory_path, f'{self.render_num}.png'), dpi=400)
            plt.close()
            self.render_num+=1


sim = simulation_env_binomial()
N_LES = sim.N_LES
N_DNS = sim.N_DNS
tolerance = 0.3
label_fontsize = 16

for filter in tqdm(sim.combined_indices):
    u_LES = np.zeros((24,sim.total_time_steps//sim.nt,N_LES//2))
    u_DNS = np.zeros((24,sim.total_time_steps//sim.nt,N_DNS//2))
    for seed in range(24):
        for i in range(sim.total_time_steps//sim.nt):
            u_LES[seed,i,:], u_DNS[seed,i,:],reward,_,_,_ = sim.step(filter)
        sim.reset(seed=seed)

    u_LES = np.mean(u_LES,axis=0)
    u_DNS = np.mean(u_DNS,axis=0)

    # Save the Mean Energy Spectrums
    directory_LES = f'../arrays/benchmarks/energy_spectrums_LES/nu_{sim.nu}_grid_{sim.N_LES}_time_steps_{sim.total_time_steps}'
    directory_DNS = f'../arrays/benchmarks/energy_spectrums_DNS/nu_{sim.nu}_grid_{sim.N_LES}_time_steps_{sim.total_time_steps}'
    filename = f'binomial_{filter}.npy'
    if not os.path.exists(directory_DNS):
        os.makedirs(directory_DNS)
    if not os.path.exists(directory_LES):
        os.makedirs(directory_LES)
    np.save(os.path.join(directory_LES, filename), u_LES)
    np.save(os.path.join(directory_DNS, filename), u_DNS)

    cum_reward = 0
    reward_storage = np.zeros(sim.total_time_steps//sim.nt)
    for render_num in range(sim.total_time_steps//sim.nt):
        reward = reward_function(u_LES[render_num,:],u_DNS[render_num,:],tolerance)
        cum_reward += reward*(0.995)**render_num

        reward_storage[render_num] = reward
    
        # plt.loglog(np.arange(N_DNS//2),u_DNS[render_num,:],label = f"N={N_DNS}")
        # plt.loglog(np.arange(N_LES//2),u_LES[render_num,:],label = f"N={N_LES}")
        # plt.loglog(np.arange(N_LES//2), reward_array(u_LES[render_num,:],u_DNS[render_num,:],tolerance), 'o', label=f"Reward", markersize=3)
        # plt.xlim([1e0,1e4])
        # plt.ylim([1e-15, 1e-1])
        # plt.xlabel("k",fontsize=label_fontsize)
        # plt.ylabel("E(k)",fontsize=label_fontsize)
        # plt.title(f"Reward={reward:.2f}, Cumulative Reward = {cum_reward:.2f}",fontsize=17)
        # plt.legend(fontsize=14,loc='lower left')        
        # # plt.show()
        # directory_path = f'../figures/benchmarks/viscosity_{sim.nu}_grid_{sim.N_LES}/binomial_{filter}/'
        # if not os.path.exists(directory_path):
        #     os.makedirs(directory_path)
        # plt.xticks(fontsize=label_fontsize)
        # plt.yticks(fontsize=label_fontsize)
        # plt.tight_layout()
        # plt.savefig(os.path.join(directory_path, f'{render_num}.png'), dpi=600)
        # plt.close()

    directory_reward = f'../arrays/benchmarks/rewards/nu_{sim.nu}_grid_{sim.N_LES}_time_steps_{sim.total_time_steps}'
    filename = f'binomial_{filter}.npy'
    if not os.path.exists(directory_reward):
        os.makedirs(directory_reward)
    np.save(os.path.join(directory_reward, filename), reward_storage)

