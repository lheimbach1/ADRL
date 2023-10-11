import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import comb

def initialize_binomial_filters(max_order):
    filters = []
    for n in range(0, max_order+1, 2):
        m = 1 / (2**n)
        d = [comb(n, k) for k in range(n+1)]
        filters.append(m * np.array(d))

    return filters

def T(k, h, alpha):
    a0 = (11/16) + (5/8) * alpha
    a1 = (15/32) + (17/16) * alpha
    a2 = (-3/16) + (3/8) * alpha
    a3 = (1/32) - (1/16) * alpha
    numerator = a0 + a1 * np.cos(h * k) + a2 * np.cos(2 * h * k) + a3 * np.cos(3 * h * k)
    denominator = 1 + 2 * alpha * np.cos(h * k)
    
    return numerator / denominator

def initialize_odd_array(length):
    middle_index = length // 2
    array = [0] * length 
    array[middle_index] = 1

    return np.array(array)



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

binomial_filters = initialize_binomial_filters(64)
smoothing_filters = compute_filters(33, 1)
combined_filters = binomial_filters + smoothing_filters
binomial_labels = {i: f"$B^{{{i*2}}}$" for i in range(len(binomial_filters))}
smoothing_labels = {i + len(binomial_filters): f"$B^{{{i + 2, 1}}}$" for i in range(len(smoothing_filters))}
combined_labels = {**binomial_labels, **smoothing_labels}

label_fontsize = 16

nu = 5e-4
N_LES = 512
time_steps = 500

directory_DNS = f'../arrays/benchmarks/energy_spectrums_DNS/nu_{nu}_grid_{N_LES}_time_steps_{time_steps}'
# Directories where the numpy arrays are saved
directory_LES = f'../arrays/benchmarks/energy_spectrums_LES/nu_{nu}_grid_{N_LES}_time_steps_{time_steps}'

files = [f for f in os.listdir(directory_LES) if f.endswith('.npy')]
integers_in_files = [(f, int(''.join(filter(str.isdigit, f)))) for f in files]
sorted_files_with_integers = sorted(integers_in_files, key=lambda x: x[1])
sorted_files = [f for f, _ in sorted(integers_in_files, key=lambda x: x[1])]

save_directory = f'../figures/report_plots/nu_{nu}_grid_{N_LES}_time_steps_{time_steps}/energy_spectrum'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

RL_data = np.load(f"../arrays/RL/nu_{nu}_grid_{N_LES}_time_steps_{time_steps}/energy_spectrum.npy")
# Plotting each array
for render_num in range(RL_data.shape[0]):
    u_DNS = np.load(os.path.join(directory_DNS, 'binomial_0.npy'))
    plt.loglog(np.arange(len(u_DNS[render_num,:])), u_DNS[render_num,:], label="DNS")
    for f, I in sorted_files_with_integers:
        linestyle = '-' if I < len(binomial_filters) else '--'

        u_LES = np.load(os.path.join(directory_LES, f))
        plt.loglog(np.arange(len(u_LES[render_num,:])), u_LES[render_num,:],linestyle=linestyle, label=combined_labels[I])
    
    plt.loglog(np.arange(len(RL_data[render_num,:])),RL_data[render_num,:],color='red',linestyle=':', label="RL")
    plt.xlabel("k", fontsize=label_fontsize)
    plt.ylabel("E(k)", fontsize=label_fontsize)
    plt.legend(fontsize=11, loc='lower left')
    plt.xticks(fontsize=label_fontsize)
    plt.yticks(fontsize=label_fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f'plot_render_{render_num}.png'), dpi=600)
    plt.clf()
