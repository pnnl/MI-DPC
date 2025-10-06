#%%
import torch; import torch.nn as nn

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
        if  self.clipping:
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



# def generate_simple_load(T, base=50, peak=1000, noise_std=0.0): # Deprecated
#     # t = torch.arange(T, dtype=torch.get_default_dtype(), requires_grad=True)
#     t = torch.arange(T)
#     hours = t / 12  # 5-min steps → 12 per hour
#     daily = 0.2 * (1 + torch.sin(2 * torch.pi * (hours - 15) / 24))
#     weekend = 1 - ((hours // 24) % 7 >= 5).float() * 0.4
#     load = base + (peak - base) * daily * weekend
#     return torch.clamp(load + torch.randn(T) * noise_std * load, base, peak)
