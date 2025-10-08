#%%
import torch
import numpy as np

class ChillerSystem(torch.nn.Module):
    def __init__(self, M, Ts, C_r, C_i, c_p, a, b, c , gamma, exponent=3, Q_rated=1000., eta_return=1., eta_supply=1.):
        super(ChillerSystem, self).__init__()
        self.M = M  # number of chillers
        self.Ts = Ts  # sampling time
        self.C_r = C_r  # Thermal capacitance return
        self.C_i = C_i  # Thermal capacitance of the chiller
        self.c_p = c_p  # Specific heat of water
        self.inv_C_i = 1.0 / self.C_i
        self.inv_C_r = 1.0 / self.C_r
        self.a = a
        self.b = b
        self.c = c
        self.gamma = gamma
        self.in_features = 1 # Number of input features for integrator
        self.out_features = self.M + 1 # Number of output features (T) for integrator
        self.exponent = exponent
        self.Q_rated = Q_rated # Rated cooling power of a chiller
        self.eta_supply = eta_supply
        self.eta_return = eta_return
    # Torch methods
    def forward_euler(self, T_supply_and_return, integer_status, mass_flow, T_evap, load, Ts=None) -> torch.Tensor:  
        """
        Inputs:
            T_return: (batch,1)        1D
            T_supply: (batch, M)      2D
            T_evap: (batch, M)        2D
            mass_flow: (batch, M)       2D
            integer_status: (batch, M)  2D
            Ts: (constant)              1D
            load: (batch,)              1D

        Outputs:
            T_return_next: (batch,)
            T_supply_next: (batch, M)
        """
        if T_supply_and_return.ndim == 2:
            T_supply = T_supply_and_return[:,:self.M]
            T_return = T_supply_and_return[:,self.M:]
        elif T_supply_and_return.ndim == 3: 
            T_supply = T_supply_and_return[:,:,:self.M]
            T_return = T_supply_and_return[:,:,self.M:]

        Ts = self.Ts if Ts is None else Ts
        
        T_supply_next = T_supply + Ts/self.C_i * self.eta_supply * (-integer_status * mass_flow * self.c_p * (T_supply - T_evap))
        temp_diff = T_return - T_supply
        energy_diff = torch.sum(self.c_p*integer_status*mass_flow*temp_diff, dim=-1, keepdim=True) 
        T_return_next = T_return + Ts/self.C_r * (load - energy_diff * self.eta_return)
        
        return torch.cat([T_supply_next, T_return_next], dim=-1)
        # return T_return_next, T_supply_next

    def exact_discretization(self, T_supply_and_return, integer_status, mass_flow, T_evap, load, Ts=None) -> torch.Tensor:
        """
        Inputs:
            T_return: (batch,1)        1D
            T_supply: (batch, M)      2D
            T_evap: (batch, M)        2D
            mass_flow: (batch, M)       2D
            integer_status: (batch, M)  2D
            Ts: (constant)              1D
            load: (batch,)              1D

        Outputs:
            T_return_next: (batch,)
            T_supply_next: (batch, M)
        """
        if T_supply_and_return.ndim == 2:
            T_supply = T_supply_and_return[:,:self.M]
            T_return = T_supply_and_return[:,self.M:]
        elif T_supply_and_return.ndim == 3: 
            T_supply = T_supply_and_return[:,:,:self.M]
            T_return = T_supply_and_return[:,:,self.M:]

        Ts = self.Ts if Ts is None else Ts

        # Calculate coefficients for T_supply_next
        coeff_supply = torch.exp(- self.eta_supply * integer_status * mass_flow * self.c_p * Ts / self.C_i)
        T_supply_next = coeff_supply * T_supply + (1 - coeff_supply) * T_evap

        # Calculate coefficients for T_return_next
        energy_coeff = torch.sum(self.eta_return * integer_status * mass_flow * self.c_p, dim=-1, keepdim=True)
        coeff_return = torch.exp(-energy_coeff * Ts / self.C_r)
        temp_term = torch.sum(self.eta_return * integer_status * mass_flow * self.c_p * T_supply, dim=-1, keepdim=True)
        T_return_next = coeff_return * T_return + (1 - coeff_return) / energy_coeff * (load + temp_term)

        # Handle cases where energy_coeff is zero to avoid division by zero
        # mask = energy_coeff == 0
        # if mask.any():
        #     T_return_next[mask] = T_return[mask] + Ts / self.C_r * load[mask]

        return torch.cat([T_supply_next, T_return_next], dim=-1)


    # def forward(self, T_supply_and_return, integer_status, mass_flow, T_evap, load, Ts=None) -> torch.Tensor: 
    def forward(self, T_supply_and_return, integer_status, mass_flow, T_evap, load, Ts=None) -> torch.Tensor: 
        """
        Inputs:
            T_return: (batch,1)        1D
            T_supply: (batch, M)      2D
            T_evap: (batch, M)        2D
            mass_flow: (batch, M)       2D
            integer_status: (batch, M)  2D
            Ts: (constant)              1D
            load: (batch,)              1D

        Outputs:
            dT_return_next: (batch,)
            dT_supply_next: (batch, M)
        """
        if T_supply_and_return.ndim == 2:
            T_supply = T_supply_and_return[:,:self.M]
            T_return = T_supply_and_return[:,self.M:]
        elif T_supply_and_return.ndim == 3: 
            T_supply = T_supply_and_return[:,:,:self.M]
            T_return = T_supply_and_return[:,:,self.M:]


        # dT_supply_next = (1/self.C_i) * (-integer_status * mass_flow * self.c_p * (T_supply - T_evap))
        # temp_diff = T_return - T_supply
        # energy_diff = torch.sum(self.c_p*integer_status*mass_flow*temp_diff, dim=-1, keepdim=True) 
        # dT_return_next = (1/self.C_r) * (load - energy_diff)

        mass_effect = self.c_p * integer_status * mass_flow
        delta_supply_evap = T_supply - T_evap
        delta_return_supply = T_return - T_supply

        dT_supply_next = self.inv_C_i * (-mass_effect * delta_supply_evap) * self.eta_supply
        energy_diff = torch.sum(
            torch.clip(mass_effect * delta_return_supply, min=0., max=self.Q_rated), 
                dim=-1, keepdim=True)
        dT_return_next = self.inv_C_r * (load - energy_diff*self.eta_return)


        return torch.cat([dT_supply_next, dT_return_next], dim=-1)
    
    def get_chiller_power_PLR(self,*, integer_status, mass_flow, T_return, T_supply) -> torch.Tensor:
        cooling = self.get_cooling_delivered_per_chiller(integer_status, mass_flow, T_return, T_supply)
        PLR = torch.clip(cooling / self.Q_rated, min=0., max=1.) 
        COP = self.a+self.b*PLR+self.c*torch.square(PLR) 
        COP = torch.relu(COP)
        power = cooling / (COP)
        power = torch.clip(power, min=0., max=self.Q_rated) # gives stable training
        return power
    
    def get_pump_consumption(self, integer_status, mass_flow) -> torch.Tensor:
        power = self.gamma * torch.pow(mass_flow, self.exponent)
        total_power = integer_status * power
        # total_power = self.gamma* torch.sum(integer_status, dim=-1, keepdim=True) * torch.pow(mass_flow, exponent)
        return total_power

    def get_cooling_delivered(self, integer_status, mass_flow, T_return, T_supply) -> torch.Tensor:
        temp_diff = T_return - T_supply
        cooling_power = torch.clip(self.c_p*integer_status*mass_flow*temp_diff, min=0., max=self.Q_rated)
        cooling_power_total =  torch.sum(cooling_power, dim=-1, keepdim=True) 
        return cooling_power_total*self.eta_return
    
    def get_cooling_delivered_per_chiller(self, integer_status, mass_flow, T_return, T_supply) -> torch.Tensor:
        # temp_diff = T_return - T_supply
        cooling_power = integer_status*self.c_p*mass_flow*(T_return - T_supply)
        return torch.clip(cooling_power*self.eta_return, min=0., max=self.Q_rated)
    
    def get_outlet_temperature(self, integer_status, mass_flow, T_supply) -> torch.Tensor:
        numerator = torch.sum(integer_status*mass_flow*T_supply, dim=-1, keepdim=True)
        denominator = torch.sum(integer_status*mass_flow, dim=-1, keepdim=True)
        return (numerator/(1e-10+denominator))
    
    
    #  NUMPY Equivalents
    def forward_np(self, integer_status, mass_flow, T_evap, T_return, T_supply, load, Ts=None) -> np.ndarray: 
        """
        Inputs:
            T_return: (batch,1)        1D
            T_supply: (batch, M)      2D
            T_evap: (batch, M)        2D
            mass_flow: (batch, M)       2D
            integer_status: (batch, M)  2D
            Ts: (constant)              1D
            load: (batch,)              1D

        Outputs:
            T_return_next: (batch,)
            T_supply_next: (batch, M)
        """
        Ts = self.Ts if Ts is None else Ts
        
        T_supply_next = T_supply + Ts/self.C_i * (-integer_status * mass_flow * self.c_p * (T_supply - T_evap))
        temp_diff = T_return - T_supply
        energy_diff = np.sum(self.c_p*integer_status*mass_flow*temp_diff, axis=-1, keepdims=True) 
        T_return_next = T_return + Ts/self.C_r * (load - energy_diff)
        return T_return_next, T_supply_next
    
    def get_pump_consumption_np(self, mass_flow) -> np.ndarray:
        total_power = np.sum(self.gamma* np.power(mass_flow, 3), axis=-1, keepdims=True)
        return total_power

    def get_cooling_delivered_np(self, integer_status, mass_flow, T_return, T_supply) -> np.ndarray:
        temp_diff = T_return - T_supply
        cooling_power = self.c_p*integer_status*mass_flow*temp_diff
        cooling_power_total =  np.sum(cooling_power, axis=-1, keepdims=True) 
        return cooling_power_total
    
    def get_outlet_temperature_np(self, integer_status, mass_flow, T_supply) -> np.ndarray:
        numerator = np.sum(integer_status*mass_flow*T_supply, axis=-1, keepdims=True)
        denominator = np.sum(integer_status*mass_flow, axis=-1, keepdims=True)
        return (numerator/(1e-10+denominator))

def kelvin2celsius(*tensors) -> torch.Tensor:
    return [tensor - 273.15 for tensor in tensors]

def celsius2kelvin(*tensors) -> torch.Tensor:
    return [tensor + 273.15 for tensor in tensors]