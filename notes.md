## Notes:

- Adding **bounded nonlinearity** after the input layer aids training stability.  
- **PWA (Piecewise Affine)** makes the control more aggressive.  
- Scaling from **kW to MW** helps with numerical instability.  

## Issues:

DPC: 
- Almost no improvement in terms of energy savings with MIDPC
- Continous decisions seem to be coupled

RBC:
- Spikes in delivered cooling when chiller turns on

## TODO:

- Implement Pyomo with continous-time dynamics and new COP computation
    - Significant improvement in energy savings?
        - Yes -> issues with DPC
        - No -> Extend the OCP
    
    - Continous decisions coupled?
        - No -> problem with DPC

- Try to run optimize at N=100; Ts=300s (Rk4 integrator required)