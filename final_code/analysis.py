#%%
import init; import torch
import matplotlib.pyplot as plt
from tabulate import tabulate

if __name__=='__main__':
    Ts = 180
    
    headers = ["Approach", "Metric", "10", "15", "20", "25", "30", "40"]

    rows = [
        ["MI-DPC (Sigmoid STE)", r"$\ell_\mathrm{mean}$", 6.7101, 4.7318, 4.0879, 3.9581, 3.8702, 3.8436],
        ["", "RSM (\\%)", "15.13\\%", "7.25\\%", "1.81\\%", "1.77\\%", "0.60\\%", "--"],
        ["", "MIT (s)\\tnote{1}", 0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004],
        ["", "NTP (-)", 82603, 84003, 85403, 86803, 88203, 91003],
        ["", "TT (s)\\tnote{2}", 585.4, 664.9, 721.9, 789.2, 862.4, 868.6],
        ["MI-DPC (Softmax STE)", r"$\ell_\mathrm{mean}$", 6.7930, 4.7851, 4.1327, 3.9483, 3.8721, 3.8656],
        ["", "RSM (\\%)", "16.55\\%", "8.45\\%", "2.93\\%", "1.53\\%", "0.65\\%", "--"],
        ["", "MIT (s)\\tnote{1}", 0.0004, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005],
        ["", "NTP (-)", 83026, 84426, 85826, 87226, 88626, 91426],
        ["", "TT (s)\\tnote{2}", 419.3, 557.8, 734.8, 589.9, 628.4, 621.5],
    ]

# Generate LaTeX table
    # latex_table = tabulate(rows, headers, tablefmt="latex")
    latex_table = tabulate(rows, headers)

    print(latex_table)
    nsteps_list=[10,20,30,40,50,60,70,80]
    # data = torch.load('results/RBC/data_N100.pt')
    # plot_chiller_data(data=data)