#%%
import torch; import torch.nn as nn;
from init import SystemParameters
import matplotlib.pyplot as plt
import matplotlib

init = SystemParameters()
# Custom MLP
class customMPL(nn.Module):
    def __init__(
        self,
        insize,
        outsize,
        hsizes=[120, 120, 120, 120],
        nonlin=nn.SELU(),
        layer_norm=False,
        affine=False,
        mins=None,
        maxs=None,
        u_min=None,
        u_max=None,
        clipping=False,
        dropout_prob=0.,
        spectral_norm=False,

    ):
        super().__init__()

        # Store normalization parameters
        if mins is None or maxs is None:
            raise ValueError("You must provide mins and maxs for each input variable.")
        if len(mins) != insize or len(maxs) != insize:
            raise ValueError("Length of mins/maxs must match number of input variables.")

        self.register_buffer("mins", torch.tensor(mins, dtype=torch.float32))
        self.register_buffer("maxs", torch.tensor(maxs, dtype=torch.float32))
        self.u_min = u_min
        self.u_max = u_max
        self.clipping = clipping

        # Build layers
        layers = []
        prev_size = insize

         # ---- Input layer ----
        linear = nn.Linear(prev_size, hsizes[0])
        if spectral_norm:
            linear = nn.utils.spectral_norm(linear)
        layers.append(linear)
        layers.append(nonlin)
        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))
        prev_size = hsizes[0]

        if layer_norm:
            layers.append(nn.LayerNorm(hsizes[0], elementwise_affine=affine))

        # ---- Hidden layers ----
        for h in hsizes[1:]:
            linear = nn.Linear(prev_size, h)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nonlin)
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            prev_size = h

        # ---- Output layer ----
        linear = nn.Linear(prev_size, outsize)
        layers.append(linear)

        self.net = nn.Sequential(*layers)

    def norm_0_1(self, x):
        denom = self.maxs - self.mins
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)  # avoid div/0
        return (x - self.mins) / denom

    def forward(self, *inputs):
        if len(inputs) > 1:
            x = torch.cat(inputs, dim=-1)
        else:
            x = inputs[0]

        # apply 0-1 normalization
        x = self.norm_0_1(x)
        out = self.net(x)
        if self.clipping:
            out = torch.clip(out, self.u_min, self.u_max)
        return out

# LOAD SIGNAL FN
def generate_datacenter_load(
sampling_time=300,   # seconds
number_of_days=1,   #  Number of days 
ramp_hours=4,       # transition duration [h]
night_baseline=350, # kW
osc_night_amp=250,  # nighttime oscillation amplitude [kW]
day_baseline=1450,  # kW
osc_day_amp=100,    # daytime oscillation amplitude [kW]
noise_scale=00,     # base random noise [kW]
ramp_jitter=00,     # extra noise during transitions [kW]
f_day=2,            # frequency day
f_night=4,          # frequency night
daily_variation=0.1, # relative variation per day
signal_seed=303,
):
    def smooth_transition(x, start, end):
        """Smooth 0→1 cosine ramp between start and end hours"""
        phase = torch.clamp((x - start) / (end - start), 0, 1)
        return 0.5 * (1 - torch.cos(torch.pi * phase))

    torch.manual_seed(signal_seed)
    # total samples for N days
    total_seconds = (24 * 3600) * number_of_days
    n_samples = int(total_seconds // sampling_time)
    t = torch.arange(n_samples) * sampling_time / 3600  # time in hours
    load = torch.zeros(n_samples)

    for d in range(number_of_days):
        day_offset = slice(d * n_samples // number_of_days,
                           (d+1) * n_samples // number_of_days)
        td = t[day_offset]  # hours for this day

        # stochastic day baselines & oscillations
        db_var = day_baseline() if callable(day_baseline) else day_baseline
        db = db_var * (1 + daily_variation * torch.rand(1).uniform_(-1,1).item())
        nb_var = night_baseline() if callable(night_baseline) else night_baseline
        nb = nb_var * (1 + daily_variation * torch.rand(1).uniform_(-1,1).item())
        oda = osc_day_amp * (1 + daily_variation * torch.rand(1).uniform_(-1,1).item())
        ona = osc_night_amp * (1 + daily_variation * torch.rand(1).uniform_(-1,1).item())

        # day/night factor
        day_factor = smooth_transition(td % 24, 12 - ramp_hours, 12) \
                   * (1 - smooth_transition(td % 24, 20, 20 + ramp_hours))
        baseline = nb + (db - nb) * day_factor

        # daily sinusoidal drift with random phase
        drift_phase = 2 * torch.pi * torch.rand(1).item()
        trend = 0.05 * baseline * torch.sin(2 * torch.pi * td / 24 + drift_phase)

        # oscillations with randomized frequencies
        freq_day = (f_day + torch.randn(1).item() * 0.5) / 24
        freq_night = (f_night + torch.randn(1).item() * 0.5) / 24
        osc_day   = oda * torch.sin(2 * torch.pi * td * freq_day)
        osc_night = ona * torch.sin(2 * torch.pi * td * freq_night)
        oscillations = day_factor * osc_day + (1 - day_factor) * osc_night

        # base random noise
        noise = torch.randn(len(td)) * noise_scale

        # extra jitter around ramp hours
        ramp_mask = ((td % 24 >= 8 - ramp_hours) & (td % 24 <= 8 + ramp_hours)) | \
                    ((td % 24 >= 20 - ramp_hours) & (td % 24 <= 20 + ramp_hours))
        transition_noise = ramp_mask.float() * torch.randn(len(td)) * ramp_jitter

        # random walk for slow drift
        random_walk = torch.cumsum(torch.randn(len(td)) * 0.05, 0)

        load[day_offset] = baseline + trend + oscillations + noise + transition_noise + random_walk
    load = torch.clamp(load, min=0)
    return t, load

def plot_chiller_data(data, save_path=None, Ts=init.Ts, time_unit=None):
    ls = ['-','--']
    clrs=['tab:blue', 'tab:red']
    s_length = data['chiller_status'].size(1)
    rng = range(data['T_evap'].size(-1))
    if time_unit=='h':
        time = torch.arange(0, data['load'].size(1), 1)*(Ts/3600) 
    elif time_unit=='s':
        time = torch.arange(0, data['load'].size(1), 1)*(Ts) 
    elif time_unit==None:
        time = torch.arange(0, data['load'].size(1), 1) 

    # # # 1) T_evap vs T_supply
    fig, axes = plt.subplots(5, 2, figsize=(18, 10))
    axes = axes.flatten()
    axes[0].plot(time, torch.ones(s_length)*init.T_supply_max, 'k--', label='Bounds')
    axes[0].plot(time, torch.ones(s_length)*init.T_supply_min, 'k--') 
    [axes[0].plot(time, 
                    data['T_supply'][0,:-1,i], 
                    label=[f"T_supply{i+1}"],
                    linestyle=ls[i//2],
                    alpha=0.7) for i in rng]

    [axes[0].plot(time, data['T_evap'][0,:,i], label=[f"T_evap{i+1}"], linestyle=ls[i//2],
                   alpha=0.7) for i in rng]
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=3); axes[0].grid(True)
    axes[0].set_xlabel("Timestep"); axes[0].set_ylabel("Temperature [°C]")

    # # # 2) Load vs Q_delivered
    axes[1].plot(time, data['load'][0,:s_length,:].cpu(), 'k--' , label="Q_demand")
    axes[1].plot(time, data['Q_delivered'][0,:,:].sum(-1, keepdim=True).cpu(), label="Q_delivered")
    axes[1].set_xlabel("Timestep"); axes[1].set_ylabel(f"Cooling [kW]")
    axes[1].legend(); axes[1].grid(True)

    # # # 3) Outlet vs retrun temperature
    axes[2].plot(time, data['T_out'][0,:,:].cpu(), label="T_out", c='b')
    axes[2].plot(time, torch.ones(s_length).cpu()*init.T_min, 'b:' ,label="T_out bounds"); 
    axes[2].plot(time, torch.ones(s_length).cpu()*init.T_max, 'b:')
    axes[2].set_xlabel("Timestep"); axes[2].set_ylabel("Temperature [°C]")
    axes[2].grid(True);    axes[2].legend()

    axes[3].plot(time, data['P_chiller'][0,:,:].cpu(), label=[f'P_chiller{i+1}' for i in rng])
    axes[3].set_xlabel("Timestep")
    axes[3].set_ylabel(f"Chiller [kW]")
    axes[3].legend()
    axes[3].grid(True)

    axes[4].plot(time, data['T_return'][0,:s_length,0].cpu(), label="T_return", c='r')
    axes[4].plot(time, torch.ones(s_length).cpu()*init.T_return_min, 'r:' ,label="T_return bounds"); 
    axes[4].plot(time, torch.ones(s_length).cpu()*init.T_return_max, 'r:'); 
    axes[4].set_xlabel("Timestep"); axes[4].set_ylabel("Temperature [°C]")
    axes[4].grid(True);    axes[4].legend()

    axes[5].plot(time, data['P_pump'][0,:,:].cpu(), label=[f'P_pump{i+1}' for i in rng])
    axes[5].grid(True);    axes[5].legend()
    axes[5].set_xlabel("Timestep")
    axes[5].set_ylabel(f"Pump [kW]")

    axes[6].plot(time, data['mass_flow'][0,:,:]*data['chiller_status'][0,:,:], label=[f'Chiller{i+1}' for i in rng])
    axes[6].plot(time, torch.ones(s_length)*0., 'k:'); axes[6].plot(time, torch.ones(s_length)*init.flow_max,'k:', label='bounds')
    axes[6].set_xlabel("Timestep")
    axes[6].set_ylabel("Mass flowrates [kg/s]")
    axes[6].legend(); axes[6].grid(True)

    axes[7].plot(time, data['chiller_status'][0,:,:], label=[f'Chiller{i+1}' for i in rng])
    try:
        axes[7].plot(time, data['relaxed_integer'][0,:,:].cpu(), '--',label=[f'Chiller{i+1} relaxed' for i in range(data['relaxed_integer'].size(-1))])
    except:
        pass
    axes[7].set_ylabel("Chiller status [-]"); axes[7].set_xlabel("Timestep")
    axes[7].legend(); axes[7].grid(True)
    
    PLR = data['Q_delivered']/(init.Q_delivered_max)
    # COP = init.a+init.b*PLR+init.c*PLR**2
    COP = data['Q_delivered'].sum(-1,keepdim=True)/data['P_chiller'].sum(-1,keepdim=True) 
    axes[8].plot(time, PLR[0,:,:].sum(-1,keepdim=True)/data['chiller_status'][0,:,:].sum(-1,keepdim=True), 
    # label=[f'Chiller{i+1}' for i in rng]
    )
    axes[8].set_ylabel("PLR [-]"); axes[8].set_xlabel("Timestep")
    # axes[8].legend()
    axes[8].grid(True)
    
    axes[9].plot(time, COP[0,:,:].mean(-1,keepdim=True), label=[f'Chiller{i+1}' for i in rng])
    axes[9].set_ylabel("COP [-]"); axes[9].set_xlabel("Timestep")
    axes[9].legend()
    axes[9].grid(True)

    n_violations = 0; tolerance = 5 # [kW]
    for i in range(s_length):
        if not data['Q_delivered'][0,i,:].sum(dim=-1, keepdim=True) + tolerance >= data['load'][0,i,0]:
            n_violations += 1
    cost = torch.sum(data['P_pump'].sum(dim=-1,keepdim=True) + data['P_chiller'].sum(dim=-1, keepdim=True) + \
                     0. *  data['chiller_status'].sum(-1,keepdim=True)) *(Ts/3600)
    
    control_RMSE = torch.sqrt(torch.mean((data['load'][:,:s_length,:] - data['Q_delivered'].sum(dim=-1, keepdim=True))**2))


    axes[1].set_title(f'Total cost of operation:  {cost.item():.1f} [kWh] \n  \
                        Tracking RMSE: {control_RMSE.item():.1f} [kW] \n \
                        Number of violations {n_violations} [-], mean COP {COP.mean():.2f}') 
    # print('Total cost of operation: ', cost.item(), 'kWh')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_chiller_data_nice(*datas, labels=None, save_path=None, Ts=300, time_unit=None):
    """
    Plot chiller data for one or more datasets on shared axes.
    Supports multiple datasets and LaTeX/PGF export.
    """
    # --- LaTeX + PGF setup ---
    matplotlib.use("pgf")
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 10,
        "pgf.rcfonts": False,
        "legend.fontsize": 6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    })

    # --- Handle colors and styles ---
    base_colors = ["royalblue", "crimson", "darkorange", "seagreen"]
    base_styles = ["-", "--", "-.", ":"]
    n_data = len(datas)
    if labels is None:
        labels = [f"Data {i+1}" for i in range(n_data)]

    # --- Time vector setup (assuming same length for all datasets) ---
    d0 = datas[0]
    s_length = d0["chiller_status"].size(1)
    if time_unit == "h":
        time = torch.arange(0, d0["load"].size(1)) * (Ts / 3600)
        x_label = "Time [h]"
    elif time_unit == "s":
        time = torch.arange(0, d0["load"].size(1)) * Ts
        x_label = "Time [s]"
    else:
        time = torch.arange(0, d0["load"].size(1))
        x_label = "Timestep [-]"

    # --- Create figure ---
    fig, axes = plt.subplots(5, 2, figsize=(7.16, 5), sharex=True)
    axes = axes.flatten()

    # --- Main plotting loop ---
    for idx, data in enumerate(datas):
        # color = base_colors[idx % len(base_colors)]
        color = None
        style = base_styles[idx % len(base_styles)]
        label_tag = labels[idx]

        # 1) T_evap vs T_supply
        for i in range(data["T_evap"].size(-1)):
            axes[0].plot(
                time, data["T_supply"][0, :-1, i],
                linestyle=style, color=color, alpha=0.8,
                label=fr"${{\mathrm{{Chiller\;{i+1}}}}}\ \mathrm{{{label_tag}}}$"
            )
            # axes[0].plot(
            #     time, data["T_evap"][0, :, i],
            #     linestyle=style, color=color, alpha=0.5,
            #     label=fr"$T_{{\mathrm{{evap,{i+1}}}}}\ \mathrm{{{label_tag}}}$"
            # )
        axes[0].plot(time, torch.ones(s_length)*init.T_min, 'k--')
        axes[0].plot(time, torch.ones(s_length)*init.T_max, 'k--')
        # 2) Load vs Q_delivered
        axes[1].plot(
            time, data["load"][0, :, :].cpu(),
            'k--',
            label=fr"$Q_\mathrm{{load}}\ \mathrm{{{label_tag}}}$"
        )
        axes[1].plot(
            time, data["Q_delivered"][0, :, :].sum(-1).cpu(),
            linestyle=style, color=color,
            label=fr"$Q\ \mathrm{{{label_tag}}}$"
        )

        # 3) T_out
        axes[2].plot(
            time, data["T_out"][0, :, :].cpu(),
            linestyle=style, color=color, alpha=0.8,
            label=fr"$T_\mathrm{{out}}\ \mathrm{{{label_tag}}}$"
        )
        axes[2].plot(time, torch.ones(s_length)*init.T_min, 'k--')
        axes[2].plot(time, torch.ones(s_length)*init.T_max, 'k--')

        # 4) P_chiller
        for i in range(data["P_chiller"].size(-1)):
            axes[3].plot(
                time, data["P_chiller"][0, :, i].cpu(),
                linestyle=style, color=color,
                label=fr"$\mathrm{{Chiller}}\;{i+1} \mathrm{{{label_tag}}}$"
            )

        # 5) T_return
        axes[4].plot(
            time, data["T_return"][0, :s_length, 0].cpu(),
            linestyle=style, color=color,
            label=fr"$T_\mathrm{{r}}\ \mathrm{{{label_tag}}}$"
        )
        axes[4].plot(time, torch.ones(s_length)*init.T_return_min, 'k--')
        axes[4].plot(time, torch.ones(s_length)*init.T_return_max, 'k--')

        # 6) P_pump
        for i in range(data["P_pump"].size(-1)):
            axes[5].plot(
                time, data["P_pump"][0, :, i].cpu(),
                linestyle=style, color=color,
                label=fr"$\mathrm{{Chiller}}\; {i+1}\ \mathrm{{{label_tag}}}$"
            )

        # 7) mass_flow
        for i in range(data["mass_flow"].size(-1)):
            axes[6].plot(
                time, data["mass_flow"][0, :, i].cpu()*data['chiller_status'][0,:,i],
                linestyle=style, color=color,
                label=fr"$\mathrm{{Chiller}}\;{i+1} \mathrm{{{label_tag}}}$"
            )
        axes[6].plot(time, torch.ones(s_length)*0., 'k--')
        axes[6].plot(time, torch.ones(s_length)*init.flow_max, 'k--')

        # 8) status
        for i in range(data["chiller_status"].size(-1)):
            axes[7].plot(
                time, data["chiller_status"][0, :, i].cpu(),
                linestyle=style, color=color,
                label=fr"$\mathrm{{Chiller}}\;{i+1}\ \mathrm{{{label_tag}}}$"
            )
        for i in range(data["relaxed_integer"].size(-1)):
            axes[7].plot(
                time, data["relaxed_integer"][0, :, i].cpu(),
                linestyle='--', color=color,
                label=fr"$\mathrm{{Chiller}}\;{i+1}\ \mathrm{{{label_tag}}}$"
            )
        # 9) PLR
        PLR = data["Q_delivered"] / init.Q_delivered_max
        axes[8].plot(
            time, PLR[0, :, :].sum(-1).cpu(),
            linestyle=style, color=color,
            label=fr"$\mathrm{{PLR}}\ \mathrm{{{label_tag}}}$"
        )

        # 10) COP
        COP = init.a + init.b * PLR + init.c * PLR ** 2
        axes[9].plot(
            time, COP[0, :, :].mean(dim=-1,keepdim=True).cpu(),
            linestyle=style, color=color,
            label=fr"$\mathrm{{COP}}\ \mathrm{{{label_tag}}}$"
        )

    # --- Formatting & labels ---
    ylabels = [
        r"$T_\mathrm{s}^{(i)}$ [°C]", r"$Q$ [kW]", r"$T_\mathrm{out}$ [°C]",
        r"$P_\mathrm{chiller}$ [kW]", r"$T_\mathrm{return}$ [°C]",
        r"$P_\mathrm{pump}$ [kW]", r"$\delta^{(i)}\dot m^{(i)}$ [kg/s]",
        r"$\delta^{(i)}$ [-]", r"PLR [-]", r"COP [-]"
    ]

    for ax, yl in zip(axes, ylabels):
        ax.set_ylabel(yl)
        ax.grid(True)
    for i in [0, 1, 3, 5, 6, 7]:
        axes[i].legend(frameon=True, framealpha=0.8, loc="best")

    axes[-1].set_xlabel(x_label)
    axes[-2].set_xlabel(x_label)
    fig.tight_layout(h_pad=0.1)
    fig.subplots_adjust(hspace=0.1)
    # --- Save ---
    if save_path is not None:
        fig.savefig(f"{save_path}.pdf", bbox_inches="tight", transparent=True, pad_inches=0.05)
        fig.savefig(f"{save_path}.pgf", bbox_inches="tight", transparent=True, pad_inches=0.05)

    plt.show()

def plot_chiller_data_paper(*datas, labels=None, save_path=None, Ts=180, time_unit=None):
    """
    Plot chiller data for one or more datasets on shared axes.
    Supports multiple datasets and LaTeX/PGF export.
    """
    # --- LaTeX + PGF setup ---
    matplotlib.use("pgf")
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 10,
        "pgf.rcfonts": False,
        "legend.fontsize": 6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    })

    # --- Handle colors and styles ---
    base_colors = ["C0", "seagreen"]
    base_styles = ["-", "--", "-.", ":"]
    n_data = len(datas)
    if labels is None:
        labels = [f"Data {i+1}" for i in range(n_data)]
    # --- Time vector setup (assuming same length for all datasets) ---
    d0 = datas[0]
    s_length = d0["chiller_status"].size(1)
    if time_unit == "h":
        time = torch.arange(0, d0["load"].size(1)) * (Ts / 3600)
        x_label = "Time [h]"
    elif time_unit == "s":
        time = torch.arange(0, d0["load"].size(1)) * Ts
        x_label = "Time [s]"
    else:
        time = torch.arange(0, d0["load"].size(1))
        x_label = "Timestep [-]"

    # --- Create figure ---
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 3.5), sharex=True)
    axes = axes.flatten()

    # --- Main plotting loop ---
    for idx, data in enumerate(datas):
        color = base_colors[idx % len(base_colors)]
        # color = None
        style = base_styles[idx % len(base_styles)]
        label_tag = labels[idx]

        
        # 5) T_return
        axes[2].plot(
            time, data["T_return"][0, :s_length, 0].cpu(),
            linestyle=style, color='crimson',
            label=fr"$T_\mathrm{{r}}$"
        )
        axes[2].plot(time, torch.ones(s_length)*init.T_return_min, 'k--')
        axes[2].plot(time, torch.ones(s_length)*init.T_return_max, 'k--')

        # 1) T_evap vs T_supply
        for i in range(data["T_evap"].size(-1)):
            axes[2].plot(
                time, data["T_supply"][0, :-1, i],
                linestyle=style, color=base_colors[i], alpha=1,
                label=fr"${{i\!=\!{i+1}}}$"
            )
            # axes[0].plot(
            #     time, data["T_evap"][0, :, i],
            #     linestyle=style, color=color, alpha=0.5,
            #     label=fr"$T_{{\mathrm{{evap,{i+1}}}}}\ \mathrm{{{label_tag}}}$"
            # )
        axes[2].plot(time, torch.ones(s_length)*init.T_min, 'k--')
        # axes[2].plot(time, torch.ones(s_length)*init.T_max, 'k--')
        # 2) Load vs Q_delivered
        axes[0].plot(
            time, data["load"][0, :, :].cpu(),
            'k--',
            label=fr"$Q_\mathrm{{load}}$"
        )
        axes[0].plot(
            time, data["Q_delivered"][0, :, :].sum(-1).cpu(),
            linestyle=style,
            label=fr"$Q$"
        )

        # 4) P_chiller
        for i in range(data["P_chiller"].size(-1)):
            axes[1].plot(
                time, data["P_chiller"][0, :, i].cpu(),
                linestyle=style, color=base_colors[i],
                label=fr"${{i\!=\!{i+1}}}$"
            )

     
        # 6) P_pump
        # for i in range(data["P_pump"].size(-1)):
        #     axes[7].plot(
        #         time, data["P_pump"][0, :, i].cpu(),
        #         linestyle=style, color=color,
        #         label=fr"$\mathrm{{Chiller}}\; {i+1}\ \mathrm{{{label_tag}}}$"
        #     )

        # 7) mass_flow
        for i in range(data["mass_flow"].size(-1)):
            axes[4].plot(
                time, data["mass_flow"][0, :, i].cpu()*data['chiller_status'][0,:,i],
                linestyle=style, color=base_colors[i],
                label=fr"${{i\!=\!{i+1}}}$"
            )
        axes[4].plot(time, torch.ones(s_length)*0., 'k--')
        axes[4].plot(time, torch.ones(s_length)*init.flow_max, 'k--')

        # 8) status

        for i in range(data["chiller_status"].size(-1)):
            axes[3].plot(
                time, data["chiller_status"][0, :, i].cpu(),
                linestyle=style, color=base_colors[i],
                label=fr"${{i\!=\!{i+1}}}$"
            )
        for i in range(data["relaxed_integer"].size(-1)):
            axes[3].plot(
                time, data["relaxed_integer"][0, :, i].cpu(),
                linestyle='--', color='darkorange',
                label=fr"$\tilde\delta$"
            )
        # 9) PLR
        # PLR = data["Q_delivered"] / init.Q_delivered_max
        # axes[8].plot(
        #     time, PLR[0, :, :].sum(-1).cpu(),
        #     linestyle=style, color=color,
        #     label=fr"$\mathrm{{PLR}}\ \mathrm{{{label_tag}}}$"
        # )

        # 10) COP
        # COP = init.a + init.b * PLR + init.c * PLR ** 2
        COP = data["Q_delivered"].sum(dim=-1, keepdim=True)/data["P_chiller"].sum(dim=-1, keepdim=True)
        axes[5].plot(
            time, 
            COP[0,:,:],
            linestyle=style, color=color,
            label=fr"$\mathrm{{COP}}\ \mathrm{{{label_tag}}}$"
        )

    # --- Formatting & labels ---
    ylabels = [ r"$Q$ [kW]", 
        r"$P_\mathrm{chiller}^{(i)}$ [kW]", 
        r"$T_\mathrm{r}$, $T_\mathrm{s}^{(i)}$ [°C]",
        r"$\tilde\delta$, $\delta^{(i)}$ [-]", 
        r"$\delta^{(i)}\dot m^{(i)}$ [kg/s]",
        r"COP [-]",
        # r"$T_\mathrm{s}^{(i)}$ [°C]" ,
        # r"$T_\mathrm{r}$ [°C]",
        # r"$P_\mathrm{pump}$ [kW]", 
        # r"PLR [-]", 
    ]

    for ax, yl in zip(axes, ylabels):
        ax.set_ylabel(yl)
        ax.grid(True)
    for i in [0, 1, 2, 3, 4]:
        axes[i].legend(frameon=True, framealpha=0.8, loc="upper left", ncol=3,
                        # bbox_to_anchor=(-0., 1.2) # on top of plot
                        bbox_to_anchor=(-0., 0.95)
                        )

    axes[-1].set_xlabel(x_label)
    axes[-2].set_xlabel(x_label)
    fig.tight_layout(h_pad=0.1, w_pad=0.2)
    fig.subplots_adjust(hspace=0.1)
    # --- Save ---
    if save_path is not None:
        fig.savefig(f"{save_path}.pdf", bbox_inches="tight", transparent=True, pad_inches=0.01)
        fig.savefig(f"{save_path}.pgf", bbox_inches="tight", transparent=True, pad_inches=0.01)

    plt.show()



# SIGNAL PLOT
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    t, load = generate_datacenter_load(sampling_time=300, number_of_days=7, ramp_hours=init.ramp_hours,
                                       f_day=5, f_night=6, 
                                       day_baseline=init.day_baseline, 
                                       night_baseline=init.night_baseline,
                                       osc_night_amp=20, osc_day_amp=20,
                                       noise_scale=5)
    
    plt.figure(figsize=(10,4)); plt.plot(t, load.numpy(), lw=1)
    plt.xlabel("Time [hours]"); plt.ylabel("Load [kW]")
    plt.grid(True, alpha=0.3); plt.show()



# def generate_simple_load(T, base=50, peak=1000, noise_std=0.0): # Deprecated
#     # t = torch.arange(T, dtype=torch.get_default_dtype(), requires_grad=True)
#     t = torch.arange(T)
#     hours = t / 12  # 5-min steps → 12 per hour
#     daily = 0.2 * (1 + torch.sin(2 * torch.pi * (hours - 15) / 24))
#     weekend = 1 - ((hours // 24) % 7 >= 5).float() * 0.4
#     load = base + (peak - base) * daily * weekend
#     return torch.clamp(load + torch.randn(T) * noise_std * load, base, peak)
