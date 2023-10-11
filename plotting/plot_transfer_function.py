import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
import numpy as np
import matplotlib.pyplot as plt
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

## PADE FILTERS
# h = 1.0
# # Create an array of k values from 0 to pi
# k_values = np.linspace(0, np.pi, 1000)
# # Iterate over alpha values and plot the transfer function for each
# alphas = np.arange(-0.5, 0.6, 0.1)
# for alpha in alphas:
#     T_values = T(k_values, h, alpha)
#     plt.plot(k_values, T_values, linestyle='--',label=f"α = {alpha:.1f}")

binomial_filters = initialize_binomial_filters(64)
smoothing_filters = compute_filters(33, 1)
combined_filters = binomial_filters + smoothing_filters

binomial_indices = [0, 1, 2,len(binomial_filters) - 1]
smoothing_indices = [len(binomial_filters), 
                    len(binomial_filters) + 1, 
                    len(binomial_filters) + len(smoothing_filters) - 1]
combined_indices = binomial_indices + smoothing_indices

binomial_labels = {i: f"$B^{{{i*2}}}$" for i in range(len(binomial_filters))}
smoothing_labels = {i + len(binomial_filters): f"$B^{{{i + 2, 1}}}$" for i in range(len(smoothing_filters))}
combined_labels = {**binomial_labels, **smoothing_labels}


for idx, i in enumerate(combined_indices):
    w, h = freqz(combined_filters[i], worN=512)
    
    # Determine linestyle based on the origin of the filter
    linestyle = '-' if i < len(binomial_filters) else '--'
    
    plt.plot(w, np.abs(h), linestyle=linestyle,  label=combined_labels[i])

label_fontsize = 16
legend_fontsize = 12

plt.xlabel("h k", fontsize=label_fontsize)
plt.ylabel("T(k)", fontsize=label_fontsize)
plt.legend(fontsize=legend_fontsize,loc="lower left")

# Increasing tick font size
plt.xticks(fontsize=label_fontsize)
plt.yticks(fontsize=label_fontsize)

plt.tight_layout()  # Adjust the layout for no clipping of labels
plt.savefig("../figures/report_plots/magnitude_response_binomial_filters.png", dpi=600)
plt.show()