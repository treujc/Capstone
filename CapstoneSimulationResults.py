import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    'font.size'           : 16.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'large',
})


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
ax1.xaxis.set_major_locator(plt.MaxNLocator(7))
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#        plt.show()
filename = 'Presentation/Eastimage4.png'
fig.savefig(filename)
plt.clf()