#%%
import torch
import numpy as np

class ChillerSystem(torch.nn.Module):
    def __init__(self, M, Ts, C_r, C_i, c_p, a, b, c , gamma):
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
    # Torch methods
    def forward_euler(self, integer_status, mass_flow, T_evap, T_return, T_supply, load, Ts=None) -> torch.Tensor: 
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
        energy_diff = torch.sum(self.c_p*integer_status*mass_flow*temp_diff, dim=-1, keepdim=True) 
        T_return_next = T_return + Ts/self.C_r * (load - energy_diff)
        return T_return_next, T_supply_next
    
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

        dT_supply_next = self.inv_C_i * (-mass_effect * delta_supply_evap)
        energy_diff = torch.sum(mass_effect * delta_return_supply, dim=-1, keepdim=True)
        dT_return_next = self.inv_C_r * (load - energy_diff)


        return torch.cat([dT_supply_next, dT_return_next], dim=-1)
    # deprecated
    # def get_chiller_consumption(self, integer_status, mass_flow, T_return, T_supply) -> torch.Tensor:
    #     cooling_delivered = self.get_cooling_delivered(integer_status, mass_flow, T_return, T_supply)
    #     power = self.a0+self.a1*cooling_delivered +self.a2*torch.square(cooling_delivered)
    #     return power
    
    # def get_chiller_power_PLR(self,*, integer_status, mass_flow, T_return, T_supply, Q_rated) -> torch.Tensor:
    #     PLR = self.get_cooling_delivered(integer_status, mass_flow, T_return, T_supply) / (torch.sum(integer_status, dim=-1, keepdim=True)*Q_rated)
    #     # PLR = self.get_cooling_delivered(integer_status, mass_flow, T_return, T_supply) / (Q_rated)
    #     COP = self.a*torch.sum(integer_status, dim=-1, keepdim=True)+self.b*PLR+self.c*torch.square(PLR) 
    #     power = self.get_cooling_delivered(integer_status, mass_flow, T_return, T_supply) / (COP+1e-10)
    #     return power

    def get_chiller_power_PLR(self,*, integer_status, mass_flow, T_return, T_supply, Q_rated) -> torch.Tensor:
        cooling = self.get_cooling_delivered_per_chiller(integer_status, mass_flow, T_return, T_supply)
        PLR = torch.clip(cooling / Q_rated, min=0., max=1.) 
        # PLR = self.get_cooling_delivered(integer_status, mass_flow, T_return, T_supply) / (Q_rated)
        # PLR = torch.clip(PLR, 0., 1.) if type(PLR) == torch.Tensor else PLR
        COP = self.a+self.b*PLR+self.c*torch.square(PLR) 
        COP = torch.relu(COP)
        # COP = torch.clip(COP, self.a, 7.) if type(COP) == torch.Tensor else COP
        power = cooling / (COP)
        power = torch.clip(power, min=0., max=Q_rated) # gives stable training
        # power = torch.clip(power, 0., Q_rated) if type(power) == torch.Tensor else power
        return power
    
    def get_pump_consumption(self, integer_status, mass_flow, exponent=3) -> torch.Tensor:
        power = self.gamma * torch.pow(mass_flow, exponent)
        total_power = integer_status * power
        # total_power = self.gamma* torch.sum(integer_status, dim=-1, keepdim=True) * torch.pow(mass_flow, exponent)
        return total_power

    def get_cooling_delivered(self, integer_status, mass_flow, T_return, T_supply) -> torch.Tensor:
        temp_diff = T_return - T_supply
        cooling_power = self.c_p*integer_status*mass_flow*temp_diff
        cooling_power_total =  torch.sum(cooling_power, dim=-1, keepdim=True) 
        return cooling_power_total
    
    def get_cooling_delivered_per_chiller(self, integer_status, mass_flow, T_return, T_supply) -> torch.Tensor:
        # temp_diff = T_return - T_supply
        cooling_power = integer_status*self.c_p*mass_flow*(T_return - T_supply)
        return cooling_power
    
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
    # deprecated
    # def get_chiller_consumption_np(self, integer_status, mass_flow, T_return, T_supply) -> np.ndarray:
        # cooling_delivered = self.get_cooling_delivered_np(integer_status, mass_flow, T_return, T_supply)
        # power = self.a0+self.a1*cooling_delivered +self.a2*np.square(cooling_delivered)
        # return power
    
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
#%%
# TESTS
if __name__ == 'main':
# if True:
    M=2
    system = ChillerSystem(M=M, Ts=1, C_r=3, C_i=2, c_p=1, gamma=5, a=1, b=1, c=1)
    print('Torch test')
    print('='*60)
    test_2d = torch.rand(100, M)
    test_1d = torch.rand(100, 1)
    test_3d = torch.rand(100, 1, 1)
    test_int = torch.bernoulli(test_2d)
    out = system.forward_euler(
                T_return=test_1d, 
                T_supply=test_2d, 
                T_evap=test_2d, 
                mass_flow=test_2d, 
                integer_status=test_int, 
                load=test_1d,
                Ts=300)

    # chiller_power = system.get_chiller_consumption(
                    # integer_status=test_int,
                    # mass_flow=test_2d,
                    # T_return=test_1d,
                    # T_supply=test_2d
    # )   
    # print(chiller_power.shape)
    
    chiller_power_PLR = system.get_chiller_power_PLR(
                    integer_status=test_int,
                    mass_flow=test_2d,
                    T_return=test_1d,
                    T_supply=test_2d, Q_rated=1000
    )   
    print(chiller_power_PLR.shape)

    pump_power = system.get_pump_consumption(
                    mass_flow=test_2d, integer=test_2d
    )   
    print(pump_power.shape)

    Q_delivered = system.get_cooling_delivered(
                    integer_status=test_int, 
                    mass_flow=test_2d,
                    T_return=test_2d,
                    T_supply=test_2d
    )   
    print(Q_delivered.shape)

    output_temperature = system.get_outlet_temperature(
                    integer_status=test_int, 
                    mass_flow=test_2d,
                    T_supply=test_2d
    )   
    print(output_temperature.shape)

    cooling_delivered = system.get_cooling_delivered(
                integer_status=test_3d,
                mass_flow=test_3d,
                T_return=test_3d,
                T_supply=test_3d
    )
    print(cooling_delivered.shape)

    print('Numpy test')
    print('='*60)

    test_2d = test_2d.numpy()    # shape: (100, M)
    test_1d = test_1d.numpy()    # shape: (100, 1)
    test_3d = test_3d.numpy()    # shape: (100, 1, 1)
    test_int = test_int.numpy()

    out_np = system.forward_np(
                T_return=test_1d, 
                T_supply=test_2d, 
                T_evap=test_2d, 
                mass_flow=test_2d, 
                integer_status=test_int, 
                load=test_1d,
                Ts=300)

    # chiller_power_np = system.get_chiller_consumption_np(
    #                 integer_status=test_int,
    #                 mass_flow=test_2d,
    #                 T_return=test_1d,
    #                 T_supply=test_2d
    # )   
    # print(chiller_power_np.shape)

    pump_power_np = system.get_pump_consumption_np(
                    mass_flow=test_2d
    )   
    print(pump_power_np.shape)

    Q_delivered_np = system.get_cooling_delivered_np(
                    integer_status=test_int, 
                    mass_flow=test_2d,
                    T_return=test_2d,
                    T_supply=test_2d
    )   
    print(Q_delivered_np.shape)

    output_temperature_np = system.get_outlet_temperature_np(
                    integer_status=test_int, 
                    mass_flow=test_2d,
                    T_supply=test_2d
    )   
    print(output_temperature_np.shape)

    cooling_delivered_np = system.get_cooling_delivered_np(
        integer_status = test_3d,
        mass_flow = test_3d,
        T_return = test_3d,
        T_supply = test_3d,
    )
    print(cooling_delivered_np.shape)


    def compare(torch_out, np_out, name):
        to_np = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
        if isinstance(torch_out, (list, tuple)):
            results = [np.allclose(to_np(t), n) for t, n in zip(torch_out, np_out)]
            for i, res in enumerate(results):
                print(f"{name}[{i}]: {'✅' if res else '❌'}")
        else:
            match = np.allclose(to_np(torch_out), np_out)
            print(f"{name:25}: {'✅' if match else '❌'}")

    compare(out, out_np, "forward")
    # compare(chiller_power, chiller_power_np, "chiller_power")
    compare(pump_power, pump_power_np, "pump_power")
    compare(Q_delivered, Q_delivered_np, "Q_delivered")
    compare(output_temperature, output_temperature_np, "output_temperature")
    compare(cooling_delivered, cooling_delivered_np, "cooling_delivered")
