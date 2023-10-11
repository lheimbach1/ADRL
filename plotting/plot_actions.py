import numpy as np
import os
import matplotlib.pyplot as plt

nu = 5e-4
N_LES = 512
time_steps = 500
label_fontsize = 16

dir_path = f'../figures/report_plots/nu_{nu}_grid_{N_LES}_time_steps_{time_steps}'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

x = np.linspace(0, time_steps, time_steps//10)
actions = np.load(f"../arrays/RL/nu_{nu}_grid_{N_LES}_time_steps_{time_steps}/actions_storage.npy")

# Compute the frequency of actions for each timestep
frequency = np.zeros((64, len(x)))
for timestep in range(actions.shape[1]):
    for action_value in range(64):
        frequency[action_value, timestep] = np.count_nonzero(actions[:, timestep] == action_value)

# Plot the heatmap
plt.imshow(frequency, aspect='auto', origin='lower', extent=[x[0], x[-1], 0, 63], cmap='viridis')
plt.plot(x, np.full(len(x), 32), color='black')
cbar = plt.colorbar()  
cbar.set_label('Count', size=label_fontsize)  
plt.xlabel("Time-Step", fontsize=label_fontsize)
plt.ylabel("Action", fontsize=label_fontsize)
plt.xticks(fontsize=label_fontsize)
plt.yticks(fontsize=label_fontsize)
plt.tight_layout()
plt.savefig(f'{dir_path}/actions_nu_{nu}_grid_{N_LES}_time_steps_{time_steps}.png', dpi=600)
plt.show()
plt.close()












# import numpy as np
# import os
# import matplotlib.pyplot as plt


# nu = 5e-4
# N_LES = 512
# label_fontsize = 16
# dir_path = f'../figures/report_plots/nu_{nu}_grid_{N_LES}_time_horizon'
# if not os.path.exists(dir_path):
#     os.makedirs(dir_path)

# x = np.linspace(0,700,70)

# actions = np.load(f"../arrays/RL/nu_{nu}_grid_{N_LES}_time_horizon/actions_storage.npy")
# plt.plot(x,np.full(len(x),32))
# for i in range(24):
#     plt.plot(x,actions[i,:],'o',markersize = '3')

# # Check if directory exists, if not then create it
# if not os.path.exists(dir_path):
#     os.makedirs(dir_path)

# plt.xlabel("Time-Step",fontsize=label_fontsize)
# plt.ylabel("Action",fontsize=label_fontsize)
# # Save the figure
# plt.xticks(fontsize=label_fontsize)
# plt.yticks(fontsize=label_fontsize)
# plt.tight_layout()

# plt.savefig(f'{dir_path}/actions_time_horizon.png', dpi=600)    
# plt.show()        
# plt.close()