#%%
import torch; import torch.nn as nn
torch.manual_seed(202)
seed = 209
Ts = 60.0 # Sampling time [s]
nsteps = 10 # Prediction horizon length

# SYSTEM PARAMETERS
M = 2 # Number of chillers
c_p = 4.184 # Specific heat of water [kJ/kgC]
fluid_per_chiller = 2000 # kg of water per chiller / ~liters
C_i = c_p * fluid_per_chiller # Thermal capacitance of chiller [kJ/C] 
fluid_in_system = 6000 # Amount of fluid in the whole system [kg] / ~[liter]
C_r = fluid_in_system*c_p # Thermal capacitance of the system [kJ/C]

# Chiller power curve coefficients with respect to PLR
a = 1.0 # kW
b = 19.33 #kW
c = -18.33 #kW
exponent = 3 # Pump power curve exponent
if exponent == 3:
    gamma = 8.625*1e-4  #8 Pump power coefficient [kWs^3/kg^3]
elif exponent == 2:
    gamma = 1.395*1e-2 # Pump power coefficient [kWs^2/kg^2]

# PROCESS BOUNDS
T_min, T_max = 8., 12. # Outlet temperature bounds [C]
T_supply_min, T_supply_max = 8., 12. # Supply temperature bounds [C]
T_evap_min, T_evap_max = 8., 12. # Evaporation temperature bounds [C]
T_return_min, T_return_max = 8., 18. # Return temperature bounds [C]
flow_min, flow_max = 5., 20. # Mass flow bounds [kg/s]
Q_delivered_max = (T_return_max - T_evap_min) * c_p * flow_max # Rated maximum cooling per chiller
Q_delivered_min = (T_return_max - T_evap_min) * c_p * flow_min
# LOAD SIGNAL FN
def generate_datacenter_load(
sampling_time=Ts,   # seconds
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
daily_variation=0.1 # relative variation per day
):
    def smooth_transition(x, start, end):
        """Smooth 0→1 cosine ramp between start and end hours"""
        phase = torch.clamp((x - start) / (end - start), 0, 1)
        return 0.5 * (1 - torch.cos(torch.pi * phase))

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
        db = day_baseline * (1 + daily_variation * torch.rand(1).uniform_(-1,1).item())
        nb = night_baseline * (1 + daily_variation * torch.rand(1).uniform_(-1,1).item())
        oda = osc_day_amp * (1 + daily_variation * torch.rand(1).uniform_(-1,1).item())
        ona = osc_night_amp * (1 + daily_variation * torch.rand(1).uniform_(-1,1).item())

        # day/night factor
        day_factor = smooth_transition(td % 24, 8 - ramp_hours, 8) \
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


import torch
import torch.nn as nn
import torch
import torch.nn as nn
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
        layers.append(nn.SELU())
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
        if not self.training and self.clipping:
            out = torch.clip(out, self.u_min, self.u_max)
        return out



# def generate_simple_load(T, base=50, peak=1000, noise_std=0.0): # Deprecated
#     # t = torch.arange(T, dtype=torch.get_default_dtype(), requires_grad=True)
#     t = torch.arange(T)
#     hours = t / 12  # 5-min steps → 12 per hour
#     daily = 0.2 * (1 + torch.sin(2 * torch.pi * (hours - 15) / 24))
#     weekend = 1 - ((hours // 24) % 7 >= 5).float() * 0.4
#     load = base + (peak - base) * daily * weekend
#     return torch.clamp(load + torch.randn(T) * noise_std * load, base, peak)

# SIGNAL AND COP PLOTS
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    t, load = generate_datacenter_load(sampling_time=45, number_of_days=1, ramp_hours=5)
    plt.figure(figsize=(10,4)); plt.plot(t, load.numpy(), lw=1)
    plt.xlabel("Time [hours]"); plt.ylabel("Load [kW]")
    plt.grid(True, alpha=0.3); plt.show()

# CHILLER POWER CHARACTERISTICS
    _PLR = torch.linspace(start=0., end=1.0, steps=100)
    _flow = torch.linspace(start=flow_min, end=flow_max, steps=100)
    pump_power_curve = gamma*_flow**exponent
    COP = (a+b*_PLR+ c*torch.square(_PLR))
    chilling_curve = Q_delivered_max*_PLR
    chiller_power_PLR_curve = chilling_curve/COP
    
    fig, axes = plt.subplots(3,2, figsize=(15,12)); axes = axes.flatten()
    axes[0].plot(_PLR,chiller_power_PLR_curve, label='P_chiller')
    axes[0].legend();axes[0].set_ylabel('P_chiller [kW]'); axes[0].set_xlabel('PLR [-]'); axes[0].grid()
     # # # 
    axes[1].plot(_PLR,COP, label='COP')
    axes[1].legend(); axes[1].set_ylabel('COP [-]'); axes[1].set_xlabel('PLR [-]'); axes[1].grid()
     # # #
    axes[2].plot(chiller_power_PLR_curve, chilling_curve)
    axes[2].set_xlabel('Chiller Power [kW]'); axes[2].set_ylabel('Q_delivered [kW]')
    axes[2].grid()
     # # #
    axes[3].plot(_PLR, chilling_curve)
    axes[3].set_xlabel('PLR [-]'); axes[3].set_ylabel('Q_delivered [kW]')
    axes[3].grid()
     # # #
    axes[4].plot(_flow, pump_power_curve)
    axes[4].set_xlabel('flow [kg/s]'); axes[4].set_ylabel('P_pump [kW]'); 
    axes[4].grid()
# %%
