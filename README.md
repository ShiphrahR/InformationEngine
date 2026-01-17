# Information Engine Simulation

This project simulates and analyzes an information engine using Langevin dynamics with feedback mechanisms. It models the behavior of a bead in a harmonic trap under various sampling frequencies and feedback rules, allowing for the study of thermodynamic efficiency and power output in information-driven systems.

## Features

- **Simulation Module**: Core simulation of bead and trap dynamics with customizable parameters
- **Analytical Calculations**: Steady-state analysis and power calculations using numerical methods
- **Feedback Analysis**: Performance analysis including efficiency, free-energy changes, and output power
- **Power Analysis**: Detailed power spectrum analysis across different frequencies

## Requirements

- Python 3.7+
- NumPy
- SciPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/information-engine.git
   cd information-engine
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The project consists of several modules that can be imported and used in your Python scripts:

### Basic Simulation

```python
from src.InformationEngine import Simulation

# Create a simulation instance
sim = Simulation(length=10000)

# Run a single trajectory
sim.single_trajectory(frequency=1.0, b_position=0, t_position=0, 
                     sigma=0, alpha=2, threshold=0, psi=0)

# Access results
bead_positions = sim.bead_positions
trap_positions = sim.traps_after
```

### Analytical Power Calculation

```python
from src.AnalyticPower import power

# Calculate power spectrum
power_values, frequencies = power(time_steps=30, delta_g=0.84)
```

### Feedback Analysis

```python
from src.FeedbackAnalysis import Create_Data, Analysor

# Create data from simulation
data = Create_Data()
# ... configure parameters ...

# Analyze the data
analyzer = Analysor(data)
efficiency = analyzer.calculate_efficiency()
```

## Project Structure

```
InformationEngine/
├── src/
│   ├── InformationEngine.py    # Core simulation classes
│   ├── AnalyticPower.py        # Analytical power calculations
│   ├── FeedbackAnalysis.py     # Feedback and efficiency analysis
│   ├── PowerAnalysis.py        # Power spectrum analysis
│   └── search_alpha.py         # Parameter optimization
├── README.md
├── LICENSE.md
├── requirements.txt
└── .gitignore
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
