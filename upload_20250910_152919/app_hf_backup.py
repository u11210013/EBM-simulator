"""
Energy Balance Model (EBM) - Hugging Face Spaces Version
氣候模擬的能量平衡模型，專為Hugging Face Spaces優化
"""

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Import immutable constants
from constants import ZONES, SUNWT, INIT_T, LAT, TCRIT, AICE, N_ZONES, get_init_a

# 設置中文字體支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def energy_balance_model_iterative(n_zones, albedo, solar_constant, emissivity, sigma, n_iter):
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
    
    Returns:
    --------
    tuple : (zone_temperatures_history, global_avg_temperatures)
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
    
    return temperatures, global_avg_temps

def run_ebm_model(n_zones, albedo, solar_constant, emissivity, sigma, n_iter, plot_types):
    """
    Run the EBM model and return results based on plot type selection
    """
    try:
        # Run the energy balance model
        zone_temps_history, global_avg_temps = energy_balance_model_iterative(
            n_zones, albedo, solar_constant, emissivity, sigma, n_iter
        )
        
        # Create DataFrame with final zone temperatures
        zone_names = list(ZONES)  # Use immutable ZONES constants for labels
        final_temps = zone_temps_history[-1, :]
        df = pd.DataFrame({
            'Zone': zone_names,
            'Final Temperature (K)': final_temps,
            'Final Temperature (°C)': final_temps - 273.15
        })
        
        # Handle multiple plot selections
        selected = plot_types if isinstance(plot_types, list) else [plot_types]
        if not selected:
            selected = ["收斂曲線 (Convergence Curve)"]
        
        # Create subplots based on number of selected visualizations
        fig, axes = plt.subplots(1, len(selected), figsize=(12 * len(selected), 8))
        if len(selected) == 1:
            axes = [axes]
        
        for ax, plot_type in zip(axes, selected):
            if plot_type == "收斂曲線 (Convergence Curve)":
                # Plot convergence curves for each zone
                iterations = range(n_iter)
                for i in range(n_zones):
                    ax.plot(iterations, zone_temps_history[:, i], label=ZONES[i], alpha=0.7)
                
                # Plot global average
                ax.plot(iterations, global_avg_temps, 'k--', linewidth=2, label='Global Average')
                
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Temperature (K)')
                ax.set_title('EBM Temperature Convergence')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
        elif plot_type == "最終溫度長條圖 (Final Temperature Bar Chart)":
            # Plot final temperatures as bar chart
            bars = ax.bar(zone_names, final_temps, alpha=0.7, color='skyblue', edgecolor='navy')
            
            # Add value labels on bars
            for bar, temp in zip(bars, final_temps):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{temp:.1f}K\n({temp-273.15:.1f}°C)',
                       ha='center', va='bottom', fontsize=9)
            
            ax.set_ylabel('Temperature (K)')
            ax.set_title('Final Zone Temperatures')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45)
            
        else:  # "溫度熱力圖 (Temperature Heatmap)"
            # Create heatmap of temperature evolution
            # Prepare data for heatmap
            heatmap_data = zone_temps_history.T  # Transpose so zones are rows, iterations are columns
            
            # Create heatmap
            im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
            
            # Set labels and title
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Zone')
            ax.set_title('Temperature Evolution Heatmap')
            
            # Set tick labels
            ax.set_xticks(range(0, n_iter, max(1, n_iter//10)))
            ax.set_xticklabels(range(0, n_iter, max(1, n_iter//10)))
            ax.set_yticks(range(n_zones))
            ax.set_yticklabels(ZONES)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Temperature (K)')
            
            # Add text annotations for final temperatures
            for i in range(n_zones):
                for j in range(0, n_iter, max(1, n_iter//5)):
                    temp = heatmap_data[i, j]
                    text_color = 'white' if temp < (np.max(heatmap_data) + np.min(heatmap_data))/2 else 'black'
                    ax.text(j, i, f'{temp:.0f}', ha='center', va='center', 
                           color=text_color, fontsize=8, weight='bold')
        
        plt.tight_layout()
        
        return df, fig
        
    except Exception as e:
        error_df = pd.DataFrame({'Error': [str(e)]})
        return error_df, None

# Create Gradio interface
with gr.Blocks(
    title="Energy Balance Model (EBM) - Climate Simulation",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    """
) as demo:
    
    gr.Markdown("""
    # 🌍 Energy Balance Model (EBM) - Climate Simulator
    
    This interactive tool simulates global climate using a simple energy balance model based on the principle that:
    
    **Absorbed Solar Radiation = Outgoing Longwave Radiation**
    
    Adjust the parameters below to explore different climate scenarios and visualize the results in multiple ways.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 🔧 Model Parameters")
            
            gr.Markdown(f"**Fixed to {N_ZONES} zones (IMMUTABLE)**")
            gr.Markdown("Zone configuration cannot be changed - using predefined climate zones")
            
            albedo = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.3, step=0.01,
                label="Albedo (Surface Reflectivity)",
                info="Fraction of solar radiation reflected back to space"
            )
            
            solar_constant = gr.Slider(
                minimum=1000, maximum=2000, value=1361, step=1,
                label="Solar Constant (W/m²)",
                info="Solar radiation intensity at Earth's distance"
            )
            
            emissivity = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.6, step=0.01,
                label="Emissivity",
                info="Surface's ability to emit thermal radiation"
            )
            
            sigma = gr.Slider(
                minimum=5e-8, maximum=6e-8, value=5.67e-8, step=1e-9,
                label="Stefan-Boltzmann Constant (W/m²/K⁴)",
                info="Physical constant for thermal radiation"
            )
            
            n_iter = gr.Slider(
                minimum=10, maximum=1000, value=100, step=10,
                label="Number of Iterations",
                info="Number of iterations for model convergence"
            )
            
            plot_types = gr.CheckboxGroup(
                choices=[
                    "收斂曲線 (Convergence Curve)", 
                    "最終溫度長條圖 (Final Temperature Bar Chart)", 
                    "溫度熱力圖 (Temperature Heatmap)"
                ],
                value=["收斂曲線 (Convergence Curve)"],
                label="Visualization Types (multi-select)",
                info="Choose 1-3 visualizations to display"
            )
            
            run_button = gr.Button("🚀 Run EBM Model", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("## 📊 Results")
            
            with gr.Tabs():
                with gr.TabItem("📈 Temperature Data"):
                    dataframe_output = gr.Dataframe(
                        headers=["Zone", "Final Temperature (K)", "Final Temperature (°C)"],
                        label="Zone Temperature Results",
                        interactive=False
                    )
                
                with gr.TabItem("📊 Visualization"):
                    plot_output = gr.Plot(label="Model Output Visualization")
    
    # Connect the button to the function - using fixed N_ZONES
    run_button.click(
        fn=lambda albedo, solar_constant, emissivity, sigma, n_iter, plot_types: run_ebm_model(N_ZONES, albedo, solar_constant, emissivity, sigma, n_iter, plot_types),
        inputs=[albedo, solar_constant, emissivity, sigma, n_iter, plot_types],
        outputs=[dataframe_output, plot_output]
    )
    
    # Add information section
    gr.Markdown("""
    ## 📚 About the Energy Balance Model
    
    ### How it works:
    1. **Solar Input**: Each zone receives solar radiation based on the solar constant
    2. **Reflection**: Some radiation is reflected back to space (albedo effect)
    3. **Absorption**: The remaining radiation is absorbed by the surface
    4. **Emission**: The surface emits thermal radiation according to Stefan-Boltzmann law
    5. **Balance**: The model iteratively finds the temperature where absorbed solar = emitted thermal radiation
    
    ### Key Parameters:
    - **Albedo (0.3)**: Typical Earth value - higher values lead to cooler temperatures
    - **Solar Constant (1361 W/m²)**: Current Earth value - represents solar intensity
    - **Emissivity (0.6)**: Surface's thermal emission efficiency
    - **Stefan-Boltzmann Constant**: Physical constant relating temperature to thermal radiation
    
    ### Visualization Options:
    - **Convergence Curve**: Shows how temperatures evolve over iterations
    - **Bar Chart**: Displays final equilibrium temperatures for each zone
    - **Heatmap**: Visualizes temperature evolution across zones and iterations
    
    ### Climate Insights:
    - Higher albedo → Cooler temperatures (more reflection)
    - Higher solar constant → Warmer temperatures (more energy input)
    - Lower emissivity → Warmer temperatures (less efficient cooling)
    """)
    
    # Add footer
    gr.Markdown("""
    ---
    **🔬 Scientific Model**: This is a simplified climate model for educational purposes. Real climate systems are much more complex and include atmospheric dynamics, ocean currents, and feedback mechanisms.
    
    **📖 Learn More**: Explore how different parameters affect global temperature and understand the basic physics of Earth's energy balance.
    """)

# Hugging Face Spaces specific configuration
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True if you want a public link
        show_error=True
    )
