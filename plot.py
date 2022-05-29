import numpy as n
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Plots/progress.csv")

# 'time/fps', 'time/total_timesteps', 'time/iterations',
# 'time/time_elapsed', 'train/value_loss', 'train/clip_range',   
# 'train/entropy_loss', 'train/explained_variance', 'train/n_updates',
# 'train/policy_gradient_loss', 'train/clip_fraction',
# 'train/learning_rate', 'train/loss', 'train/approx_kl'

plt.plot(data['train/value_loss'].values)
plt.show()
