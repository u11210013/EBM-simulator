---
title: Energy Balance Model (EBM) - Climate Simulator
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.44.1
app_file: app_hf.py
pinned: false
license: mit
short_description: Interactive climate simulation using energy balance model
---

# 🌍 Energy Balance Model (EBM) - Climate Simulator

An interactive climate simulation tool that models global temperature using energy balance principles. This application allows you to explore how different climate parameters affect global temperature through an intuitive web interface.

## 🔬 What is an Energy Balance Model?

An Energy Balance Model (EBM) is a simplified climate model based on the fundamental principle that:

**Absorbed Solar Radiation = Outgoing Longwave Radiation**

This model helps us understand how Earth's temperature is determined by the balance between incoming solar energy and outgoing thermal radiation.

## 🚀 Features

- **Interactive Parameter Adjustment**: Modify climate parameters in real-time
- **Multiple Visualizations**: 
  - Convergence curves showing temperature evolution
  - Bar charts displaying final temperatures
  - Heatmaps visualizing temperature patterns across zones and time
- **Educational Interface**: Learn about climate physics through hands-on experimentation
- **Real-time Results**: Instant feedback on parameter changes

## 🎛️ Adjustable Parameters

- **Number of Climate Zones**: 1-20 zones for spatial resolution
- **Albedo**: Surface reflectivity (0.0-1.0)
- **Solar Constant**: Solar radiation intensity (1000-2000 W/m²)
- **Emissivity**: Surface thermal emission efficiency (0.0-1.0)
- **Stefan-Boltzmann Constant**: Physical constant for thermal radiation
- **Iterations**: Number of convergence steps (10-1000)

## 📊 Visualization Options

1. **Convergence Curve**: Shows how temperatures evolve over iterations
2. **Final Temperature Bar Chart**: Displays equilibrium temperatures for each zone
3. **Temperature Heatmap**: Visualizes temperature evolution across zones and iterations

## 🧪 Educational Use Cases

- **Climate Science Education**: Understand basic climate physics
- **Parameter Sensitivity**: Explore how different factors affect global temperature
- **Model Validation**: Compare results with known Earth conditions
- **Hypothetical Scenarios**: Test "what-if" climate scenarios

## 🔧 Technical Details

- **Model Type**: Zero-dimensional energy balance model
- **Physics**: Stefan-Boltzmann law for thermal radiation
- **Numerical Method**: Iterative relaxation for convergence
- **Interface**: Gradio web application
- **Visualization**: Matplotlib and Seaborn

## 📚 Scientific Background

The model implements the following physics:

1. **Solar Input**: Each zone receives solar radiation based on the solar constant
2. **Reflection**: Some radiation is reflected back to space (albedo effect)
3. **Absorption**: The remaining radiation is absorbed by the surface
4. **Emission**: The surface emits thermal radiation according to Stefan-Boltzmann law
5. **Balance**: The model finds the temperature where absorbed solar = emitted thermal radiation

## 🌡️ Default Earth-like Parameters

- **Albedo**: 0.3 (typical Earth value)
- **Solar Constant**: 1361 W/m² (current Earth value)
- **Emissivity**: 0.6 (typical for Earth's surface)
- **Stefan-Boltzmann Constant**: 5.67×10⁻⁸ W/m²/K⁴

## 🎯 Learning Objectives

After using this tool, you should understand:

- How Earth's temperature is determined by energy balance
- The role of albedo in climate regulation
- How solar intensity affects global temperature
- The relationship between temperature and thermal radiation
- Basic principles of climate modeling

## ⚠️ Important Notes

This is a **simplified educational model** and should not be used for:
- Climate predictions
- Policy decisions
- Scientific research without additional validation

Real climate systems are much more complex and include:
- Atmospheric dynamics
- Ocean currents
- Feedback mechanisms
- Seasonal variations
- Regional differences

## 🔗 Related Resources

- [Energy Balance Models in Climate Science](https://en.wikipedia.org/wiki/Energy_balance_model)
- [Stefan-Boltzmann Law](https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law)
- [Earth's Energy Budget](https://en.wikipedia.org/wiki/Earth%27s_energy_budget)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Happy Climate Modeling! 🌍📊**
