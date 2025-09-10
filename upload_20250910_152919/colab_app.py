import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Import immutable constants
from constants import ZONES, SUNWT, INIT_T, LAT, TCRIT, AICE, N_ZONES, get_init_a

# 確保在Colab環境中正確導入
try:
    from ebm import energy_balance_model_iterative
except ImportError:
    # 如果無法導入，直接定義函數
    def energy_balance_model_iterative(n_zones, albedo, solar_constant, emissivity, sigma, n_iter, precision=2):
        """
        Energy Balance Model (EBM) iterative solver
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

def run_ebm_model(n_zones, albedo, solar_constant, emissivity, sigma, n_iter, plot_types, precision=2):
    """
    Run the EBM model and return results based on plot type selection
    """
    try:
        # Run the energy balance model
        zone_temps_history, global_avg_temps = energy_balance_model_iterative(
            n_zones, albedo, solar_constant, emissivity, sigma, n_iter, precision
        )
        
        # Create DataFrame with final zone temperatures
        zone_names = list(ZONES)  # Use immutable ZONES constants for labels
        final_temps = zone_temps_history[-1, :]
        df = pd.DataFrame({
            'Zone': zone_names,
            'Final Temperature (K)': final_temps,
            'Final Temperature (°C)': final_temps - 273.15
        }).round(int(precision))
        
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
                           f'{temp:.{int(precision)}f}K\n({temp-273.15:.{int(precision)}f}°C)',
                           ha='center', va='bottom', fontsize=9)
                
                ax.set_ylabel('Temperature (K)')
                ax.set_title('Final Zone Temperatures')
                ax.grid(True, alpha=0.3, axis='y')
                plt.setp(ax.get_xticklabels(), rotation=45)
                
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
                        ax.text(j, i, f'{temp:.{int(precision)}f}', ha='center', va='center', 
                               color=text_color, fontsize=8, weight='bold')
        
        plt.tight_layout()
        
        return df, fig
        
    except Exception as e:
        error_df = pd.DataFrame({'Error': [str(e)]})
        return error_df, None

# Create Gradio interface
with gr.Blocks(title="Energy Balance Model (EBM) - Colab Version") as demo:
    gr.Markdown("# Energy Balance Model (EBM) Simulator")
    gr.Markdown("This tool simulates global climate using a simple energy balance model.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Model Parameters")
            
            gr.Markdown(f"**Fixed to {N_ZONES} zones (IMMUTABLE)**")
            gr.Markdown("Zone configuration uses predefined immutable constants")
            
            albedo = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.3, step=0.01,
                label="Albedo (Surface Reflectivity)"
            )
            
            solar_constant = gr.Slider(
                minimum=1000, maximum=2000, value=1361, step=1,
                label="Solar Constant (W/m²)"
            )
            
            emissivity = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.6, step=0.01,
                label="Emissivity"
            )
            
            sigma = gr.Slider(
                minimum=5e-8, maximum=6e-8, value=5.67e-8, step=1e-9,
                label="Stefan-Boltzmann Constant (W/m²/K⁴)"
            )
            
            n_iter = gr.Slider(
                minimum=10, maximum=1000, value=100, step=10,
                label="Number of Iterations"
            )
            
            precision = gr.Slider(
                minimum=1, maximum=10, value=2, step=1,
                label="Display Precision (Decimal Places)",
                info="Number of decimal places to display in results"
            )
            
            plot_types = gr.CheckboxGroup(
                choices=["收斂曲線 (Convergence Curve)", "最終溫度長條圖 (Final Temperature Bar Chart)", "溫度熱力圖 (Temperature Heatmap)"],
                value=["收斂曲線 (Convergence Curve)"],
                label="Plot Types (multi-select)",
                info="Choose 1-3 visualizations to display"
            )
            
            run_button = gr.Button("Run EBM Model", variant="primary")
        
        with gr.Column():
            gr.Markdown("## Results")
            
            dataframe_output = gr.Dataframe(
                headers=["Zone", "Final Temperature (K)", "Final Temperature (°C)"],
                label="Zone Temperature Results"
            )
            
            plot_output = gr.Plot(label="Model Output Plot")
    
    # Connect the button to the function - using fixed N_ZONES
    run_button.click(
        fn=lambda albedo, solar_constant, emissivity, sigma, n_iter, plot_types, precision: run_ebm_model(N_ZONES, albedo, solar_constant, emissivity, sigma, n_iter, plot_types, precision),
        inputs=[albedo, solar_constant, emissivity, sigma, n_iter, plot_types, precision],
        outputs=[dataframe_output, plot_output]
    )
    
    # Add some example parameters
    gr.Markdown("""
    ## Example Parameters:
    - **Default values** are set to reasonable Earth-like conditions
    - **Albedo**: 0.3 (typical Earth value)
    - **Solar Constant**: 1361 W/m² (current Earth value)
    - **Emissivity**: 0.6 (typical for Earth's surface)
    - **Stefan-Boltzmann Constant**: 5.67×10⁻⁸ W/m²/K⁴
    """)

# Colab專用啟動方式
if __name__ == "__main__":
    # 在Colab中啟動，使用share=True產生公開連結
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
