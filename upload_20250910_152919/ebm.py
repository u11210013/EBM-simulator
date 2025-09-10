import numpy as np

def energy_balance_model_iterative(n_zones, albedo, solar_constant, emissivity, sigma, n_iter, precision=2):
    """
    Energy Balance Model (EBM) iterative solver
    
    Parameters:
    -----------
    n_zones : int
        Number of climate zones
    albedo : float
        Surface albedo (reflectivity)
    solar_constant : float
        Solar constant (W/m²)
    emissivity : float
        Surface emissivity
    sigma : float
        Stefan-Boltzmann constant (W/m²/K⁴)
    n_iter : int
        Number of iterations
    precision : int
        Number of decimal places for rounding results
    
    Returns:
    --------
    tuple : (zone_temperatures_history, global_avg_temperatures)
        - zone_temperatures_history: array of shape (n_iter, n_zones) with temperature for each zone at each iteration
        - global_avg_temperatures: array of shape (n_iter,) with global average temperature at each iteration
    """
    
    # Initialize temperature array for all zones
    temperatures = np.zeros((n_iter, n_zones))
    global_avg_temps = np.zeros(n_iter)
    
    # Initial temperature guess (in Kelvin)
    initial_temp = 288.0  # ~15°C
    temperatures[0, :] = initial_temp
    global_avg_temps[0] = initial_temp
    
    # Solar flux per zone (assuming equal distribution)
    solar_flux = solar_constant / 4.0  # Divide by 4 for spherical Earth
    
    # Iterative solution
    for i in range(1, n_iter):
        # Calculate outgoing longwave radiation for each zone
        outgoing_lw = emissivity * sigma * temperatures[i-1, :]**4
        
        # Calculate absorbed solar radiation for each zone
        absorbed_solar = solar_flux * (1 - albedo)
        
        # Energy balance: absorbed solar = outgoing longwave
        # Solve for temperature: T = (absorbed_solar / (emissivity * sigma))^(1/4)
        new_temps = (absorbed_solar / (emissivity * sigma))**(1/4)
        
        # Apply relaxation factor for stability (0.1 = 10% of new value, 90% of old value)
        relaxation_factor = 0.1
        temperatures[i, :] = (1 - relaxation_factor) * temperatures[i-1, :] + relaxation_factor * new_temps
        
        # Calculate global average temperature
        global_avg_temps[i] = np.mean(temperatures[i, :])
    
    # Round results to specified precision
    temperatures = np.round(temperatures, int(precision))
    global_avg_temps = np.round(global_avg_temps, int(precision))
    
    return temperatures, global_avg_temps
