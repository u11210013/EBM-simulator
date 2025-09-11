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
    def energy_balance_model_iterative(aice, frac_sc, sc, K, tcrit, B, A, a, n_iter, precision=2):
        """
        Energy Balance Model (EBM) iterative solver
        """
        from decimal import Decimal, getcontext
        
        # Set precision
        getcontext().prec = 50
        
        # Convert parameters to Decimal
        aice = Decimal(str(aice))
        frac_sc = Decimal(str(frac_sc))
        sc = Decimal(str(sc))
        K = Decimal(str(K))
        tcrit = Decimal(str(tcrit))
        B = Decimal(str(B))
        A = Decimal(str(A))
        a = Decimal(str(a))
        
        # Use constants from constants.py
        sunwt = [Decimal(str(sw)) for sw in SUNWT]
        init_t = [Decimal(str(t)) for t in INIT_T]
        lat = [Decimal(str(l)) for l in LAT]
        
        final_t = [t for t in init_t]
        final_a = [aice if t < tcrit else a for t in init_t]
        all_temps = []
        
        tolerance = Decimal("0.0000000001")
        
        for iteration in range(n_iter):
            Tcos = [final_t[i] * Decimal(np.cos(np.radians(float(lat[i])))) for i in range(N_ZONES)]
            mean_temp = sum(Tcos) / sum(Decimal(np.cos(np.radians(float(l)))) for l in lat)
            
            new_t = []
            new_a = []
            r_in_list = [(sc / Decimal("4")) * frac_sc * sw for sw in sunwt]
            
            for i in range(N_ZONES):
                r_in_i = r_in_list[i]
                temp_i = (r_in_i * (Decimal("1") - final_a[i]) + K * mean_temp - A) / (B + K)
                albedo_i = aice if temp_i < tcrit else a
                new_t.append(temp_i)
                new_a.append(albedo_i)
            
            # Check convergence
            converged = True
            for i in range(N_ZONES):
                if abs(new_t[i] - final_t[i]) > tolerance:
                    converged = False
                    break
            
            final_t = new_t
            final_a = new_a
            all_temps.append([float(t) for t in final_t])
            
            if converged:
                break
        
        # Convert to numpy arrays for compatibility
        temperatures = np.array(all_temps)
        global_avg_temps = np.array([np.mean(row) for row in temperatures])
        
        # Determine convergence status
        convergence_status = f"收斂於第 {iteration + 1} 次迭代" if converged else "未平衡"
        
        return temperatures, global_avg_temps, convergence_status

def run_ebm_model(aice, frac_sc, sc, K, tcrit, B, A, a, n_iter, plot_types, precision=2):
    """
    Run the EBM model and return results based on plot type selection
    """
    try:
        # Run the energy balance model
        zone_temps_history, global_avg_temps, convergence_status = energy_balance_model_iterative(
            aice, frac_sc, sc, K, tcrit, B, A, a, n_iter, precision
        )
        
        # Create DataFrame with final zone temperatures
        zone_names = list(ZONES)  # Use immutable ZONES constants for labels
        final_temps = zone_temps_history[-1, :]
        df = pd.DataFrame({
            'Zone': zone_names,
            'Final Temperature (°C)': final_temps,
            'Convergence Status': [convergence_status] * len(zone_names)
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
                for i in range(N_ZONES):
                    ax.plot(iterations, zone_temps_history[:, i], label=ZONES[i], alpha=0.7)
                
                # Plot global average
                ax.plot(iterations, global_avg_temps, 'k--', linewidth=2, label='Global Average')
                
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Temperature (°C)')
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
                
                # Set tick labels
                actual_iterations = len(zone_temps_history)
                ax.set_xticks(range(0, actual_iterations, max(1, actual_iterations//10)))
                ax.set_xticklabels(range(0, actual_iterations, max(1, actual_iterations//10)))
                ax.set_yticks(range(N_ZONES))
                ax.set_yticklabels(ZONES)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Temperature (°C)')
                
                # Add text annotations for final temperatures
                actual_iterations = len(zone_temps_history)
                for i in range(N_ZONES):
                    for j in range(0, actual_iterations, max(1, actual_iterations//5)):
                        if j < heatmap_data.shape[1]:
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
            
            aice = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.6, step=0.01,
                label="aice (Ice Albedo)"
            )
            
            frac_sc = gr.Slider(
                minimum=0.1, maximum=2.0, value=1.0, step=0.01,
                label="frac_sc (Solar Fraction)"
            )
            
            sc = gr.Slider(
                minimum=1000, maximum=2000, value=1370, step=1,
                label="sc (Solar Constant W/m²)"
            )
            
            K = gr.Slider(
                minimum=1.0, maximum=10.0, value=3.87, step=0.01,
                label="K (Heat Transport)"
            )
            
            tcrit = gr.Slider(
                minimum=-20, maximum=0, value=-10, step=1,
                label="tcrit (Critical Temperature °C)"
            )
            
            B = gr.Slider(
                minimum=1.0, maximum=5.0, value=2.17, step=0.01,
                label="B (Radiation Parameter)"
            )
            
            A = gr.Slider(
                minimum=100, maximum=300, value=204, step=1,
                label="A (Radiation Constant)"
            )
            
            a = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.3, step=0.01,
                label="a (Surface Albedo)"
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
                headers=["Zone", "Final Temperature (°C)", "Convergence Status"],
                label="Zone Temperature Results"
            )
            
            plot_output = gr.Plot(label="Model Output Plot")
    
    # Connect the button to the function
    run_button.click(
        fn=run_ebm_model,
        inputs=[aice, frac_sc, sc, K, tcrit, B, A, a, n_iter, plot_types, precision],
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
