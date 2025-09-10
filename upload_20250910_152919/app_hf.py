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
from decimal import Decimal, getcontext

# Import immutable constants
from constants import ZONES, SUNWT, INIT_T, LAT, TCRIT, AICE, N_ZONES, get_init_a

# 設置中文字體支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def energy_balance_model_iterative(max_iterations=500, tolerance=Decimal("0.0000000001")):
    """
    Energy Balance Model using the exact calculation method provided
    """
    
    # Set precision
    getcontext().prec = 50
    
    # Constants from provided code
    aice = Decimal("0.6")
    frac_sc = Decimal("1")
    sc = Decimal("1370")
    K = Decimal("3.87")
    tcrit = Decimal("-10")
    B = Decimal("2.17")
    A = Decimal("204")
    a = Decimal("0.3")
    
    # Data from constants
    zones = ["90-80", "80-70", "70-60", "60-50", "50-40", "40-30", "30-20", "20-10", "10-0"]
    sunwt = [Decimal("0.5"), Decimal("0.531"), Decimal("0.624"), Decimal("0.77"), Decimal("0.892"), Decimal("1.021"), Decimal("1.12"), Decimal("1.189"), Decimal("1.219")]
    init_t = [Decimal("-15"), Decimal("-15"), Decimal("-5"), Decimal("5"), Decimal("10"), Decimal("15"), Decimal("18"), Decimal("22"), Decimal("24")]
    lat = [Decimal("85"), Decimal("75"), Decimal("65"), Decimal("55"), Decimal("45"), Decimal("35"), Decimal("25"), Decimal("15"), Decimal("6")]
    init_a = [aice if t < tcrit else a for t in init_t]
    
    final_t = [t for t in init_t]
    final_a = [a for a in init_a]
    all_temps = []
    global_mean_temp_final = Decimal("0")
    all_tcos = []
    
    for iteration in range(max_iterations):
        Tcos = [final_t[i] * Decimal(np.cos(np.radians(float(lat[i])))) for i in range(len(zones))]
        mean_temp = sum(Tcos) / sum(Decimal(np.cos(np.radians(float(l)))) for l in lat)
        all_tcos.append(Tcos)
        
        new_t = []
        new_a = []
        r_in_list = [(sc / Decimal("4")) * frac_sc * sw for sw in sunwt]
        r_out_list = []
        f_i_list = []
        r_in_1_a_list = []
        balance_list = []
        
        for i in range(len(zones)):
            r_in_i = r_in_list[i]
            r_out_i = A + B * final_t[i]
            temp_i = (r_in_i * (Decimal("1") - final_a[i]) + K * mean_temp - A) / (B + K)
            albedo_i = aice if temp_i < tcrit else a
            init_a[i] = albedo_i
            new_t.append(temp_i)
            new_a.append(albedo_i)
            r_out_list.append(r_out_i)
            f_i_i = K * (temp_i - mean_temp)
            f_i_list.append(f_i_i)
            r_in_1_a = r_in_i * (Decimal("1") - final_a[i])
            r_in_1_a_list.append(r_in_1_a)
            balance = r_out_i + f_i_i - r_in_1_a
            balance_list.append(balance)
        
        # Check convergence
        converged = True
        for i in range(len(zones)):
            if abs(new_t[i] - final_t[i]) > tolerance:
                converged = False
                break
        
        final_t = new_t
        final_a = new_a
        all_temps.append(final_t[:])
        global_mean_temp_final = mean_temp
        
        if converged:
            break
    
    return final_t, final_a, global_mean_temp_final, all_temps, r_in_list, r_out_list, f_i_list, all_tcos, r_in_1_a_list, balance_list, init_a

def run_ebm_model(n_zones, albedo, solar_constant, emissivity, sigma, n_iter, plot_types, precision=2):
    """
    Run the EBM model using the exact calculation method provided
    """
    try:
        # Run the energy balance model with exact calculation
        final_t, final_a, global_mean_temp_final, all_temps, r_in_list, r_out_list, f_i_list, all_tcos, r_in_1_a_list, balance_list, init_a_final = energy_balance_model_iterative(
            max_iterations=n_iter, tolerance=Decimal("0.0000000001")
        )
        
        # Use the exact data from the calculation
        zones = ["90-80", "80-70", "70-60", "60-50", "50-40", "40-30", "30-20", "20-10", "10-0"]
        sunwt = [Decimal("0.5"), Decimal("0.531"), Decimal("0.624"), Decimal("0.77"), Decimal("0.892"), Decimal("1.021"), Decimal("1.12"), Decimal("1.189"), Decimal("1.219")]
        init_t = [Decimal("-15"), Decimal("-15"), Decimal("-5"), Decimal("5"), Decimal("10"), Decimal("15"), Decimal("18"), Decimal("22"), Decimal("24")]
        lat = [Decimal("85"), Decimal("75"), Decimal("65"), Decimal("55"), Decimal("45"), Decimal("35"), Decimal("25"), Decimal("15"), Decimal("6")]
        
        # Create DataFrame with proper formatting and shorter column names
        df = pd.DataFrame({
            'Zone': zones,
            'Lat': [round(float(l), int(precision)) for l in lat],
            'SunWt': [round(float(sw), int(precision)) for sw in sunwt],
            'Init_T': [round(float(t), int(precision)) for t in init_t],
            'Init_a': [round(float(a), int(precision)) for a in init_a_final],
            'Final_T': [round(float(t), int(precision)) for t in final_t],
            'Final_a': [round(float(a), int(precision)) for a in final_a],
            'R_in': [round(float(r), int(precision)) for r in r_in_list],
            'R_out': [round(float(r), int(precision)) for r in r_out_list],
            'R_in(1-a)': [round(float(r), int(precision)) for r in r_in_1_a_list],
            'Balance:R_out+F(i)=R_in*(1-a)': [round(float(b), int(precision)) for b in balance_list],
            'F_i': [round(float(f), int(precision)) for f in f_i_list],
            'T_cos': [round(float(t), int(precision)) for t in all_tcos[-1]]
        })
        
        # Convert all_temps to numpy array for plotting
        zone_temps_history = np.array([[float(t) for t in row] for row in all_temps])
        
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
                actual_iterations = len(zone_temps_history)
                iterations = range(actual_iterations)
                for i in range(len(zones)):
                    ax.plot(iterations, zone_temps_history[:, i], label=zones[i], alpha=0.7)
                
                # Calculate global average for plotting
                global_avg_for_plot = [np.mean(row) for row in zone_temps_history]
                ax.plot(iterations, global_avg_for_plot, 'k--', linewidth=2, label='Global Average')
                
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Temperature (K)')
                ax.set_title('EBM Temperature Convergence')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            elif plot_type == "最終溫度長條圖 (Final Temperature Bar Chart)":
                # Plot final temperatures as bar chart
                final_temps_celsius = [float(t) for t in final_t]
                bars = ax.bar(zones, final_temps_celsius, alpha=0.7, color='skyblue', edgecolor='navy')
                
                # Add value labels on bars
                for bar, temp in zip(bars, final_temps_celsius):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{temp:.{int(precision)}f}°C',
                           ha='center', va='bottom', fontsize=9)
                
                ax.set_ylabel('Temperature (°C)')
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
                
                # Set tick labels using actual data dimensions
                actual_iterations = len(zone_temps_history)
                ax.set_xticks(range(0, actual_iterations, max(1, actual_iterations//10)))
                ax.set_xticklabels(range(0, actual_iterations, max(1, actual_iterations//10)))
                ax.set_yticks(range(len(zones)))
                ax.set_yticklabels(zones)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Temperature (°C)')
                
                # Add text annotations for final temperatures
                for i in range(len(zones)):
                    for j in range(0, actual_iterations, max(1, actual_iterations//5)):
                        if j < heatmap_data.shape[1]:  # Check bounds
                            temp = heatmap_data[i, j]
                            text_color = 'white' if temp < (np.max(heatmap_data) + np.min(heatmap_data))/2 else 'black'
                            ax.text(j, i, f'{temp:.{int(precision)}f}', ha='center', va='center', 
                                   color=text_color, fontsize=8, weight='bold')
        
        plt.tight_layout()
        
        # Use the exact global temperature calculation from provided code
        global_temp_display = f"全球平均溫度: {float(global_mean_temp_final):.{int(precision)}f}°C"
        
        # Create iteration DataFrame with iteration number column
        iter_df = pd.DataFrame(zone_temps_history, columns=zones).round(int(precision))
        # Add iteration number as the first column
        iter_df.insert(0, '迭代次數', range(1, len(iter_df) + 1))
        iter_df.index.name = 'Row'
        
        return df, fig, global_temp_display, iter_df
        
    except Exception as e:
        error_df = pd.DataFrame({'Error': [str(e)]})
        return error_df, None, f"錯誤: {str(e)}", pd.DataFrame()

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
        with gr.Column(scale=3):
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
            
            precision = gr.Slider(
                minimum=1, maximum=10, value=2, step=1,
                label="Display Precision (Decimal Places)",
                info="Number of decimal places to display in results"
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
        
        with gr.Column(scale=7):
            gr.Markdown("## 📊 Results")
            
            # Global Temperature Display Block
            with gr.Group():
                gr.Markdown("### 全球平均溫度")
                global_temp_output = gr.Markdown(value="全球平均溫度: -- °C", elem_classes=["global-temp"])
            
            # Results in Two Columns Layout
            with gr.Row():
                with gr.Column(scale=1):
                    # Results Table Block
                    with gr.Group():
                        gr.Markdown("### 完整結果表格")
                        dataframe_output = gr.Dataframe(
                            label="Energy Balance Results", 
                            interactive=False,
                            wrap=False
                        )
                
                with gr.Column(scale=1):
                    # Iteration Process Block
                    with gr.Group():
                        gr.Markdown("### 迭代過程")
                        iter_output = gr.Dataframe(
                            label="Temperature Evolution by Iteration", 
                            interactive=False,
                            wrap=False
                        )
            
            # Visualization Block - Full Width
            with gr.Group():
                gr.Markdown("### 可視化圖表")
                plot_output = gr.Plot(label="Selected Visualizations")
    
    # Connect the button to the function - parameters are now handled internally
    run_button.click(
        fn=lambda albedo, solar_constant, emissivity, sigma, n_iter, plot_types, precision: run_ebm_model(9, albedo, solar_constant, emissivity, sigma, n_iter, plot_types, precision),
        inputs=[albedo, solar_constant, emissivity, sigma, n_iter, plot_types, precision],
        outputs=[dataframe_output, plot_output, global_temp_output, iter_output]
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
