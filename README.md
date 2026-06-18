# EA Sensor Optimization Project

An end-to-end Evolutionary Computing project that optimizes IoT sensor placement in a smart city environment using Genetic Algorithms (GA), Differential Evolution (DE), and a Hybrid GA-DE approach.

The system analyzes traffic-density data, transforms it into a spatial grid, and automatically discovers sensor locations that maximize coverage while minimizing overlap and uncovered high-priority regions.

---

## Project Overview

Smart city deployments often require placing a limited number of sensors while ensuring maximum monitoring efficiency.

This project formulates sensor placement as an optimization problem and applies Evolutionary Algorithms to identify near-optimal sensor configurations.

The solution includes:

- Data preprocessing and grid generation
- Coverage simulation
- Fitness-based optimization
- Genetic Algorithm implementation
- Differential Evolution implementation
- Hybrid GA-DE optimization
- Interactive Streamlit dashboard
- Experimental comparison framework

---

## Features

### Data Processing
- Load and preprocess traffic sensor datasets
- Convert geographical coordinates into a grid representation
- Generate weighted heatmaps based on traffic importance

### Optimization Algorithms
- Genetic Algorithm (GA)
- Differential Evolution (DE)
- Hybrid GA-DE Algorithm

### Advanced GA Components
- Tournament Selection
- Roulette Wheel Selection
- Stochastic Universal Sampling (SUS)
- Over-selection
- One-Point Crossover
- Uniform Crossover
- Random Shift Mutation
- Random Reset Mutation
- Diversity Preservation Strategies
- Elitism and Rank-Based Survivor Selection

### Visualization
- Grid Visualization
- Traffic Heatmaps
- Density Maps
- High-Traffic Region Analysis
- Fitness Evolution Curves
- Sensor Placement Visualization

### Interactive Dashboard
Built with Streamlit for:

- Parameter tuning
- Algorithm selection
- Real-time experimentation
- Comparative analysis

---

## System Architecture

```text
Traffic Dataset
       │
       ▼
Data Cleaning & Processing
       │
       ▼
Grid Generation
       │
       ▼
Fitness Evaluation
       │
       ▼
Evolutionary Algorithms
   • Genetic Algorithm
   • Differential Evolution
   • Hybrid GA-DE
       │
       ▼
Optimal Sensor Locations
       │
       ▼
Visualization & Analysis
```

---

## Optimization Objective

The optimization process aims to:

### Maximize
- Coverage of important traffic regions
- Monitoring effectiveness

### Minimize
- Sensor overlap
- Uncovered critical areas
- Redundant sensor placement

---

## Algorithms Implemented

### Genetic Algorithm (GA)

#### Selection Methods
- Tournament Selection
- Roulette Wheel Selection
- Stochastic Universal Sampling (SUS)
- Over Selection

#### Crossover Operators
- One-Point Crossover
- Uniform Crossover

#### Mutation Operators
- Random Shift Mutation
- Random Reset Mutation

#### Diversity Preservation
- Basic Diversity Control
- Strong Diversity Control

#### Survivor Selection
- Elitism
- Rank-Based Replacement

### Differential Evolution (DE)

Implemented components:

- DE Mutation
- DE Crossover
- Adaptive Mutation Factor (F)
- Adaptive Crossover Rate (CR)
- Solution Repair Mechanisms

### Hybrid GA-DE

Combines the strengths of both algorithms:

- GA performs exploration
- DE performs refinement
- Improved convergence
- Better search-space coverage
- Enhanced solution quality

---

## Project Structure

```text
EA-Sensor-Optimization-Project/
│
├── core/
├── ui/
├── data/
├── main.py
├── config.py
├── requirements.txt
├── EA_Code_Ready_for_UI.ipynb
├── IoT_Sensor_EA_Project_Experiments.ipynb
└── Smart_City_IoT_Report.docx
```

---

## Installation

```bash
git clone https://github.com/Jomana-ElSaghier/EA-Sensor-Optimization-Project.git

cd EA-Sensor-Optimization-Project

pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run main.py
```

---

## Experimental Framework

The project supports experimentation across:

- Selection methods
- Crossover methods
- Mutation methods
- Initialization methods
- Survivor strategies
- Different sensor counts
- GA, DE, and Hybrid approaches

---

## Evaluation Metrics

- Coverage Score
- Coverage Ratio
- Overlap Penalty
- Uncovered Area Penalty
- Population Diversity
- Fitness Progression
- Convergence Behavior

---

## Team Contributions

### Shahd AbdElhay — Problem Modeling, Dataset Processing and Grid Design

Responsibilities:

- Dataset loading and preprocessing
- Feature extraction
- Data normalization
- Geographic grid generation
- Problem formulation

### Jana Ahmed — Fitness Function and Coverage Simulation

Responsibilities:

- Coverage modeling
- Coverage-area calculation
- Overlap detection
- Data importance weighting
- Fitness function design

### Jomana El-Saghier — Genetic Algorithm Core

Responsibilities:

- Population initialization
- Parent selection strategies
- Survivor selection strategies
- Population management

### Evan — Genetic Operators and Diversity Control

Responsibilities:

- Crossover operators
- Mutation operators
- Diversity preservation
- Parameter tuning

### Adam Fadel — Differential Evolution, Hybrid System, UI and Experiments

Responsibilities:

- Differential Evolution implementation
- Hybrid GA-DE integration
- Experimental evaluation
- Streamlit interface development
- Final system integration

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Joblib
- Evolutionary Computation Techniques

---

## Future Improvements

- Multi-objective optimization
- Dynamic sensor deployment
- Real-time traffic adaptation
- Parallel optimization execution
- Reinforcement learning integration
- Larger smart-city datasets

---

## Contributors

| Member   | Name               | GitHub                                             | LinkedIn                                                       |
|----------|--------------------|----------------------------------------------------|----------------------------------------------------------------|
| Member 1 | Shahd AbdElhay     | [GitHub](https://github.com/shahd-abdelhay)        | [LinkedIn](https://www.linkedin.com/in/shahd-abdelhay-3538022a2/) |
| Member 2 | Jana Ahmed         | [GitHub](https://github.com/janaahmeed)            | [LinkedIn](https://www.linkedin.com/in/jana-ahmed-mostafa-1216972a5/) |
| Member 3 | Jomana El-Saghier  | [GitHub](https://github.com/Jomana-ElSaghier)      | [LinkedIn](https://www.linkedin.com/in/jomana-el-saghier-a602512ba/) |
| Member 4 | Evan               | [GitHub](https://github.com/)                      | [LinkedIn](https://www.linkedin.com/in/) |
| Member 5 | Adam Fadel         | [GitHub](https://github.com/AdamFadel)             | [LinkedIn](https://www.linkedin.com/in/adam-fadel/) |

---

---

## License

This project was developed for educational and research purposes.

---

## Acknowledgments

Special thanks to all project contributors for their work on evolutionary optimization, smart-city analytics, system integration, experimentation, and visualization development.
