import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal, getcontext

# Import immutable constants from dedicated constants module
from constants import ZONES, SUNWT, INIT_T, LAT, TCRIT, AICE, N_ZONES, get_init_a

def energy_balance_model_iterative_decimal(aice, frac_sc, sc, Kc, tcrit, Bc, Ac, a, max_iterations):
    # Initialize with immutable constants - parameters cannot be modified
    final_t = [t for t in INIT_T]  # Use immutable INIT_T constant
    # Calculate initial albedo using immutable helper function
    final_a = list(get_init_a(a))  # Use immutable constants via helper function
    r_in_list = [(sc / Decimal("4")) * frac_sc * sw for sw in SUNWT]
    all_temps, all_tcos = [], []

    for _ in range(int(max_iterations)):
        Tcos = [final_t[i] * Decimal(np.cos(np.radians(float(LAT[i])))) for i in range(N_ZONES)]
        mean_temp = sum(Tcos) / sum(Decimal(np.cos(np.radians(float(l)))) for l in LAT)
        all_tcos.append(Tcos)

        new_t, new_a = [], []
        r_out_list, f_i_list, r_in_1_a_list, balance_list = [], [], [], []

        for i in range(N_ZONES):  # Use immutable N_ZONES constant
            r_in_i = r_in_list[i]
            temp_i = (r_in_i * (Decimal("1") - final_a[i]) + Kc * mean_temp - Ac) / (Bc + Kc)
            albedo_i = AICE if temp_i < TCRIT else a  # Use immutable constants

            r_out_i = Ac + Bc * temp_i
            f_i_i = Kc * (temp_i - mean_temp)
            r_in_1_a = r_in_i * (Decimal("1") - albedo_i)
            balance = r_out_i + f_i_i - r_in_1_a

            new_t.append(temp_i); new_a.append(albedo_i)
            r_out_list.append(r_out_i); f_i_list.append(f_i_i)
            r_in_1_a_list.append(r_in_1_a); balance_list.append(balance)

        final_t, final_a = new_t, new_a
        all_temps.append(final_t[:])

    return final_t, final_a, mean_temp, all_temps, r_in_list, r_out_list, f_i_list, all_tcos, r_in_1_a_list, balance_list, final_a


def run_ebm_model(aice, frac_sc, sc, Kc, tcrit, Bc, Ac, a, n_iter, precision, plot_types):
    try:
        getcontext().prec = max(int(precision) + 5, 20)

        (final_t, final_a, global_mean_temp_final, all_temps, r_in_list, r_out_list,
         f_i_list, all_tcos, r_in_1_a_list, balance_list, init_a_final) = energy_balance_model_iterative_decimal(
            Decimal(str(aice)), Decimal(str(frac_sc)), Decimal(str(sc)), Decimal(str(Kc)),
            Decimal(str(tcrit)), Decimal(str(Bc)), Decimal(str(Ac)), Decimal(str(a)), int(n_iter)
        )

        zone_temps_history = np.array([[float(x) for x in row] for row in all_temps])
        
        # Calculate energy balance results with all required columns
        results_df = pd.DataFrame({
            'Zones': ZONES,
            'Lat': [float(LAT[i]) for i in range(N_ZONES)],
            'SunWt': [float(SUNWT[i]) for i in range(N_ZONES)],
            'Init_T': [float(INIT_T[i]) for i in range(N_ZONES)],
            'Init_a': [float(get_init_a()[i]) for i in range(N_ZONES)],
            'Final_T': [float(final_t[i]) for i in range(N_ZONES)],
            'Final_a': [float(final_a[i]) for i in range(N_ZONES)],
            'R_in': [float(x) for x in r_in_list],
            'R_out': [float(x) for x in r_out_list],
            'R_in*(1-a)': [float(x) for x in r_in_1_a_list],
            'Balance:R_out+F(i)=R_in*(1-a)': [float(x) for x in balance_list],
            'F(i)': [float(x) for x in f_i_list],
            'Tcos': [float(final_t[i] * Decimal(np.cos(np.radians(float(LAT[i]))))) for i in range(N_ZONES)]
        }).round(int(precision))
        global_avg_temps = np.array([float(sum(row)/len(row)) for row in zone_temps_history])
        final_vals = zone_temps_history[-1, :]
        zone_names = list(ZONES)  # Use immutable ZONES constants for labels

        selected = plot_types if isinstance(plot_types, list) else [plot_types]
        if not selected:
            selected = ["收斂曲線 (Convergence Curve)"]

        fig, axes = plt.subplots(1, len(selected), figsize=(12 * len(selected), 7))
        if len(selected) == 1:
            axes = [axes]

        for ax, ptype in zip(axes, selected):
            if ptype == "收斂曲線 (Convergence Curve)":
                iters = range(len(zone_temps_history))
                for i in range(N_ZONES):
                    ax.plot(iters, zone_temps_history[:, i], alpha=0.7, label=ZONES[i])
                ax.plot(iters, global_avg_temps, 'k--', lw=2, label='Global Avg')
                ax.set_xlabel('Iteration'); ax.set_ylabel('Temperature (°C)')
                ax.set_title('Convergence'); ax.legend(); ax.grid(True, alpha=0.3)
            elif ptype == "最終溫度長條圖 (Final Temperature Bar Chart)":
                bars = ax.bar(zone_names, final_vals, color='skyblue', edgecolor='navy')
                for bar, v in zip(bars, final_vals):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f'{v:.1f}°C', ha='center', va='bottom', fontsize=9)
                ax.set_ylabel('Temperature (°C)'); ax.set_title('Final Zone Temperatures')
                ax.grid(True, axis='y', alpha=0.3); plt.setp(ax.get_xticklabels(), rotation=45)
            elif ptype == "溫度熱力圖 (Temperature Heatmap)":
                im = ax.imshow(zone_temps_history.T, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
                ax.set_xlabel('Iteration'); ax.set_ylabel('Zone'); ax.set_title('Temperature Heatmap (°C)')
                ax.set_yticks(range(N_ZONES)); ax.set_yticklabels(ZONES)
                ax.set_xticks(range(0, len(zone_temps_history), max(1, len(zone_temps_history)//10)))
                ax.set_xticklabels(range(0, len(zone_temps_history), max(1, len(zone_temps_history)//10)))
                cbar = plt.colorbar(im, ax=ax); cbar.set_label('°C')

        plt.tight_layout()

        # Calculate final global equilibrium temperature
        cos_weights = [np.cos(np.radians(float(LAT[i]))) for i in range(N_ZONES)]
        weighted_temp_sum = sum(zone_temps_history[-1, i] * cos_weights[i] for i in range(N_ZONES))
        total_weight = sum(cos_weights)
        global_equilibrium_temp = weighted_temp_sum / total_weight
        
        summary = f"Final Global Equilibrium Temperature: {global_equilibrium_temp:.{int(precision)}f} °C"
        iter_df = pd.DataFrame(zone_temps_history, columns=zone_names).round(int(precision))
        iter_df.index.name = 'Iteration'

        return results_df, fig, summary, iter_df

    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]}), None, f"Error: {str(e)}", pd.DataFrame()

# Create Gradio interface
with gr.Blocks(title="Energy Balance Model (EBM)") as demo:
    gr.Markdown("# Energy Balance Model (EBM) Simulator")
    gr.Markdown("This tool simulates global climate using a simple energy balance model.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"## Model Parameters (Zones fixed to {N_ZONES} - IMMUTABLE)")
            gr.Markdown("**Note: Zone configuration, latitudes, solar weights, and initial temperatures are immutable constants**")
            aice = gr.Number(value=0.6, label="aice", info="Ice albedo (matches immutable AICE constant)")
            frac_sc = gr.Number(value=1.0, label="frac_sc")
            sc = gr.Number(value=1370, label="sc (Solar Constant)")
            Kc = gr.Number(value=3.87, label="K")
            tcrit = gr.Number(value=-10, label="tcrit (°C)", info="Critical temperature (matches immutable TCRIT constant)")
            Bc = gr.Number(value=2.17, label="B")
            Ac = gr.Number(value=204, label="A")
            a = gr.Number(value=0.3, label="a (surface albedo)")
            n_iter = gr.Slider(minimum=10, maximum=1000, value=100, step=10, label="Iterations")
            precision = gr.Slider(minimum=2, maximum=10, value=2, step=1, label="Display precision (decimals)")
            plot_types = gr.CheckboxGroup(
                choices=["收斂曲線 (Convergence Curve)", "最終溫度長條圖 (Final Temperature Bar Chart)", "溫度熱力圖 (Temperature Heatmap)"],
                value=["收斂曲線 (Convergence Curve)"],
                label="Plot Types (multi-select)"
            )
            run_button = gr.Button("Run EBM Model", variant="primary")
            show_iter_button = gr.Button("Show Full Iteration Log")
        
        with gr.Column():
            gr.Markdown("## Results")
            with gr.Tabs():
                with gr.TabItem("Table"):
                    dataframe_output = gr.Dataframe(label="Full Results Table", interactive=False)
                with gr.TabItem("Visualization"):
                    plot_output = gr.Plot(label="Selected Visualizations")
                with gr.TabItem("Summary"):
                    summary_output = gr.Markdown(label="Summary")
                with gr.TabItem("Iterations"):
                    iter_output = gr.Dataframe(label="Iteration Temperatures", interactive=False)
    
    # Connect the button to the function
    run_button.click(
        fn=run_ebm_model,
        inputs=[aice, frac_sc, sc, Kc, tcrit, Bc, Ac, a, n_iter, precision, plot_types],
        outputs=[dataframe_output, plot_output, summary_output, iter_output]
    )

    show_iter_button.click(
        fn=run_ebm_model,
        inputs=[aice, frac_sc, sc, Kc, tcrit, Bc, Ac, a, n_iter, precision, plot_types],
        outputs=[dataframe_output, plot_output, summary_output, iter_output]
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

if __name__ == "__main__":
    demo.launch()
