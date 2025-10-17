#%%
import torch; import torch.nn as nn
from chiller_system import ChillerSystem
class SystemParameters():
    def __init__(self, exponent=3, Ts=180, M=2):
        self.seed = 209
        self.Ts = Ts # Sampling time [s]

        self.M = M # Number of chillers

        self.Q_delivered_max = 500 # rated chiller cooling [kW] - This value is provided by the manufacturer

        # # # Load signal parameters
        self.night_baseline=lambda: torch.rand(1).uniform_(100,350)
        self.day_baseline=lambda: torch.rand(1).uniform_(300,(self.M*self.Q_delivered_max)*0.75)
        self.ramp_hours = 4
        self.tolerance = 5 # tolerance for cooling bound [kW]
        self.chiller_on_cost = 10.
        # SYSTEM PARAMETERS
        self.c_p = 4.184 # Specific heat of water [kJ/kgC]
        self.fluid_per_chiller = 3500 # kg of water per chiller / ~liters
        self.C_i = self.c_p * self.fluid_per_chiller # Thermal capacitance of chiller [kJ/C] 
        self.fluid_in_system = 7000 # Amount of fluid in the whole system [kg] / ~[liter]
        self.C_r = self.fluid_in_system*self.c_p # Thermal capacitance of the system [kJ/C]
        self.P0 = 10  # Power penalty for having chiller on
        self.load_filter = [0.45,0.2,0.15,0.1, 0.05, 0.05]
        # load_filter = [1.]
        self.eta_supply=0.7 # Heat transition efficiency coefficients (chiller)
        self.eta_return=0.75 # Heat transition efficiency coefficients (cooling end)

        # Chiller power curve coefficients with respect to PLR
        self.a = 1.0 # kW
        self.b = 19.33 #kW
        self.c = -18.33 #kW
        self.exponent = 3 # Pump power curve exponent
        if exponent == 3:
            self.gamma = 9.625*1e-4  #8 Pump power coefficient [kWs^3/kg^3]
        elif exponent == 2:
            self.gamma = 1.395*1e-2 # Pump power coefficient [kWs^2/kg^2]

        # PROCESS BOUNDS
        self.T_min, self.T_max = 8., 12. # Outlet temperature bounds [C]
        self.T_supply_min, self.T_supply_max = 8., 12. # Supply temperature bounds [C]
        self.T_evap_min, self.T_evap_max = 8., 12. # Evaporation temperature bounds [C]
        self.T_return_min, self.T_return_max = 8., 40. # Return temperature bounds [C]
        self.flow_min, self.flow_max = 5., 20. # Mass flow bounds [kg/s]


# Q_delivered_max = (T_return_max - T_evap_min) * c_p * flow_max # Rated maximum cooling per chiller
# Q_delivered_min = (T_return_max - T_evap_min) * c_p * flow_min

# delta_penalty = 1. # penalty coefficient for chiller status switching

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    init = SystemParameters()
# CHILLER POWER CHARACTERISTICS
    _PLR = torch.linspace(start=0., end=1., steps=100)
    _flow = torch.linspace(start=init.flow_min, end=init.flow_max, steps=100)
    pump_power_curve = init.gamma*_flow**init.exponent
    COP = (init.a+init.b*_PLR+ init.c*torch.square(_PLR))
    chilling_curve = init.Q_delivered_max*_PLR
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
