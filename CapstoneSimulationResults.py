import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


current_dir = os.path.realpath(".")
data_folder = os.path.join(current_dir,'data')
datafile = os.path.join(data_folder,'SimulationResults.csv')

df = pd.read_csv(datafile, low_memory = False)

figure_size_l = 8.5
figure_size_w = figure_size_l/.77
fig, ax1 = plt.subplots(figsize= (figure_size_w,figure_size_l))
plt.title('East BU Simulation Run')
color = 'red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Incidents', color=color)
ax1.plot(df['DateKey'], df['Rolling14Forward'], 'o', color=color, label='Forward Rolling 14 Day Incident Count')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='best')
color = 'darkgreen'
ax1.plot(df['DateKey'], df['Prediction'], color=color, label='Model Prediction')
ax1.legend(loc='best') 
fig.autofmt_xdate()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#        plt.show()
filename = 'Presentation/Eastimage4.png'
fig.savefig(filename)
plt.clf()