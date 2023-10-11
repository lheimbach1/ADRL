import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags, csc_matrix
import os
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
def reward_function(e_LES, e_DNS, tolerance):
    e_DNS_trimmed = e_DNS[:len(e_LES)]
    
    # Calculate relative errors
    relative_errors = np.zeros(len(e_LES))
    valid_indices = e_DNS_trimmed != 0
    relative_errors[valid_indices] = np.abs(e_LES[valid_indices] - e_DNS_trimmed[valid_indices]) / np.abs(e_DNS_trimmed[valid_indices])
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
    rewards = np.where(relative_errors < tolerance, 1e-2, 5e-15)    
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


class discrete_filters(gym.Env):
    def __init__(self):

        #Current and Total Number of Time-Steps per Episode
        self.time_step = 0
        self.total_time_steps = 500
        #Viscosity
        self.nu = 5e-4
        #Timestep
        self.dt = 1e-4
        #Number of Time-Steps between Filter Update
        self.nt = 10
        #Declaration Grid Sizes of LES and DNS
        self.N_LES = 512
        self.N_DNS = int(self.N_LES*4)

        #Initalization of Binomial and Binomial Smoothing Filters
        binomial_filters = initialize_binomial_filters(64)
        smoothing_filters = compute_filters(33, 1)
        self.combined_filters = binomial_filters + smoothing_filters

        #Declare State and Action space for the Environment
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.N_LES,),dtype=np.float64)
        self.action_space = spaces.Discrete(len(self.combined_filters))

        dx = 2*np.pi/self.N_DNS
        k = np.fft.fftfreq(self.N_DNS,dx)*2*np.pi
        random_dict = {abs_val: np.random.uniform(0, 1) for abs_val in np.abs(k)}
        self.u_DNS, self.dx_DNS, self.k_DNS, dk = initalize_paper(self.N_DNS,random_dict)
        self.u_LES, self.dx_LES, self.k_LES, dk = initalize_paper(self.N_LES,random_dict)      

        #Approximate Deconvolution Parameters
        self.beta = 1.
        self.Q = 2
        
        #Relative Percentage Error Tolerance in Reward Function
        self.tolerance = 0.3

        #Variables to Save Behavior of RL Model During Testing
        self.cumulative_reward = 0
        self.num_episodes = 24
        self.episode_counter = 0
        self.render_num=0
        self.step_counter = 0
        self.steps_per_episode = self.total_time_steps//self.nt
        self.e_LES_ensemble = np.zeros((self.steps_per_episode, self.num_episodes, self.N_LES//2))
        self.e_DNS_ensemble = np.zeros((self.steps_per_episode, self.num_episodes, self.N_DNS//2))
        self.rewards_storage = np.zeros(self.total_time_steps//self.nt)
        self.action_storage = np.zeros((24,self.total_time_steps//self.nt))

    def step(self,action):
        G = self.combined_filters[action]
        
        #Saving actions for behavioral anaylsis of RL model
        self.action = action
        self.action_storage[self.episode_counter-1,self.step_counter] = action

        for t in range(self.nt):
            self.u_DNS = tvdrk3_burgers(self.u_DNS, self.dt, self.dx_DNS, self.nu)
            self.u_LES = tvdrk3_burgers_explicit_filter(self.u_LES, self.dt, self.dx_LES, self.nu, self.beta, self.Q, G)
            self.time_step += 1
        
        reward = reward_function(compute_energy_spectrum(self.u_LES), compute_energy_spectrum(self.u_DNS),self.tolerance)

        if self.time_step == self.total_time_steps:
            done = True
            truncated = True
        else: 
            done = False
            truncated = False

        info = {}
        self.step_counter+=1
        return self.u_LES, reward, done, truncated, info

    def reset(self,seed=None,options=None):
        np.random.seed(seed)
        self.time_step = 0
        dx = 2*np.pi/self.N_DNS
        k = np.fft.fftfreq(self.N_DNS,dx)*2*np.pi
        random_dict = {abs_val: np.random.uniform(0, 1) for abs_val in np.abs(k)}
        self.u_DNS, self.dx_DNS, self.k_DNS, dk = initalize_paper(self.N_DNS,random_dict)
        self.u_LES, self.dx_LES, self.k_LES, dk = initalize_paper(self.N_LES,random_dict)  
        info = {}

        #Counters to Save Behavior of RL Model during Evaluation
        self.render_num = 0
        self.cumulative_reward = 0
        self.episode_counter += 1
        self.step_counter = 0

        return self.u_LES, info
    

    #Rendering Function for Ensemble Averaged Results
    def render(self, mode='human'):
        if mode == 'human':
            print("Episode: ", self.episode_counter, "Render: ", self.render_num)
            e_LES = compute_energy_spectrum(self.u_LES)
            e_DNS = compute_energy_spectrum(self.u_DNS)
            
            self.e_LES_ensemble[self.render_num, self.episode_counter-1] = e_LES
            self.e_DNS_ensemble[self.render_num, self.episode_counter-1] = e_DNS

            if self.render_num == (self.total_time_steps//self.nt)-1 and self.episode_counter == self.num_episodes:
                #Create Directory to Save Actions and Ensemble Averaged Energy Spectrum
                arrays_path = f"../arrays/RL/nu_{self.nu}_grid_{self.N_LES}_time_steps_{self.total_time_steps}"
                if not os.path.exists(arrays_path):
                        os.makedirs(arrays_path)

                #Creat Directory to Save Plots of Ensemble Averaged Energy Spectrum
                dir_path = f'../figures/discrete_filters_energy_spectrum_ensemble_average/nu_{self.nu}_grid_{self.N_LES}_time_steps_{self.total_time_steps}/'
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                
                #Save Actions and Ensemble Averaged Energy Spectrum
                np.save(f"{arrays_path}/actions_storage.npy",self.action_storage)
                np.save(f"{arrays_path}/energy_spectrum.npy",np.mean(self.e_LES_ensemble,axis=1))

                label_fontsize = 16
                #Plot Ensemble Averaged Energy Spectrum every 10 time-steps
                for i in range(self.total_time_steps//self.nt):
                    avg_e_LES = np.mean(self.e_LES_ensemble[i], axis=0)
                    avg_e_DNS = np.mean(self.e_DNS_ensemble[i], axis=0)

                    reward = reward_function(avg_e_LES, avg_e_DNS, self.tolerance)
                    self.rewards_storage[i] = reward
                    self.cumulative_reward += reward*(0.995)**i
                    # Plotting
                    plt.loglog(self.k_DNS[:self.N_DNS // 2], avg_e_DNS[:self.N_DNS // 2], label=f"N={self.N_DNS}")
                    plt.loglog(self.k_LES[:self.N_LES // 2], avg_e_LES[:self.N_LES // 2], label=f"N={self.N_LES}")
                    plt.loglog(np.arange(self.N_LES//2), reward_array(avg_e_LES, avg_e_DNS, self.tolerance), 'o', label=f"Reward", markersize=3)
                    # plt.ylim([1e-15, 1e-1])
                    plt.xlabel("k",fontsize=label_fontsize)
                    plt.ylabel("E(k)",fontsize=label_fontsize)
                    plt.title(f"Reward={reward:.2f}, Cumulative Reward={self.cumulative_reward:.2f}",fontsize=17)
                    plt.legend(fontsize=14,loc='lower left')
                    plt.xticks(fontsize=label_fontsize)
                    plt.yticks(fontsize=label_fontsize)
                    plt.tight_layout()
                    plt.savefig(f'{dir_path}/{i}.png', dpi=600)            
                    plt.close()

                #Save the Rewards every 10 time-steps of the Ensemble Averaged Energy Spectrum
                np.save(f"{arrays_path}/rewards.npy",self.rewards_storage)
            self.render_num += 1

    # #Rendering Function for One Episode
    # def render(self, mode='human'):
    #     if mode == 'human':
    #         print("Render Number: ", self.render_num)
    #         #Creat Directory to Save Plots of Energy Spectrum
    #         dir_path = f'../figures/discrete_filters_energy_spectrum/viscosity_{self.nu}_grid_{self.N_LES}_time_steps_{self.total_time_steps}/'
    #         if not os.path.exists(dir_path):
    #             os.makedirs(dir_path)

    #         e_LES = compute_energy_spectrum(self.u_LES)
    #         e_DNS = compute_energy_spectrum(self.u_DNS)
    #         reward = reward_function(e_LES,e_DNS,self.tolerance)
    #         self.cumulative_reward += reward*(0.995)**self.render_num
    #         plt.loglog(self.k_DNS[:self.N_DNS // 2],e_DNS,label = f"N={self.N_DNS}")
    #         plt.loglog(self.k_LES[:self.N_LES // 2],e_LES,label = f"N={self.N_LES}")
    #         plt.loglog(np.arange(self.N_LES//2), reward_array(e_LES,e_DNS,self.tolerance), 'o', label=f"Percentage Reward", markersize=3)
    #         # plt.ylim([1e-20, 1e-1])
    #         plt.xlabel("k")
    #         plt.ylabel("E(k)")
    #         plt.title(f"Reward={reward}, Cum. Reward={self.cumulative_reward}, Action={self.action}")
    #         plt.legend()
    #         plt.savefig(f'{dir_path}/{self.render_num}.png', dpi=600)            
    #         plt.close()
    #         self.render_num+=1

gym.envs.register(
    id='Simulation-predict_discrete_filters',
    entry_point='environment:discrete_filters'
)