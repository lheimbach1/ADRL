import pandas as pd
import matplotlib.pyplot as plt

label_fontsize = 16
df = pd.read_csv('../arrays/training_rewards.csv')
plt.plot(df.iloc[:, 0], df.iloc[:, 1])
plt.xlabel(df.columns[0],fontsize=label_fontsize)
plt.ylabel("Cumulative Reward",fontsize=label_fontsize)
plt.xticks(fontsize=label_fontsize)
plt.yticks(fontsize=label_fontsize)
plt.tight_layout()
plt.savefig("../figures/report_plots/training_reward.png", dpi=600)
plt.show()
