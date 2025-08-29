# Star Formation History Analysis

WARNING: THIS README IS GENERATED USING CLAUDE CODE AND CONTAINS SOME HALLUCINATIONS. THESE WILL BE CLEANED UP ONCE REFACTORING IS DONE.

Repository containing the data and code used in **Savelli & Kustec et al. (in prep.)** - "*Classifying Galaxy Star Formation Histories with Machine Learning*".

## Overview

This repository provides a comprehensive analysis of star formation histories (SFHs) from multiple galaxy simulations using machine learning techniques. The analysis includes:

- **UMAP dimensionality reduction** of normalized SFHs
- **Simulation of Origin Probabilities** (SOPs) for classifying galaxies by their parent simulation
- **Autoencoder analysis** for pattern recognition in SFH data  
- **Milky Way Analogue** (MWA) identification and analysis
- **Comprehensive visualization tools** including "battleship grid" plots

## Repository Structure

```bash
savelli-kustec-gal-sfh/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data/                        # Data directory
│   ├── Iyer_etal_2020/         # SFH data from Iyer et al. 2020
│   └── autoencoder_results/     # Autoencoder prediction results  
├── src/                         # Source code
│   └── utils/                   # Utility modules
│       ├── analysis.py          # Core analysis functions
│       ├── plotting.py          # Plotting utilities
│       └── data_processing.py   # Data loading and processing
└── notebooks/                   # Analysis notebooks
    ├── 01_data_preparation.ipynb    # Data loading and preprocessing
    ├── 02_main_sequence.ipynb       # Galaxy main sequence (Figure 1)
    ├── 03_sfh_analysis.ipynb        # SFH sample plots (Figure 2)  
    ├── 04_umap_analysis.ipynb       # UMAP analysis (Figures 3-5)
    ├── 05_sop_analysis.ipynb        # Simulation origin probabilities (Figures 6-7)
    ├── 06_autoencoder.ipynb         # Autoencoder results
    └── 07_mwa_analysis.ipynb        # Milky Way analogues
```

## Installation

### Requirements

- Python 3.8+
- See `requirements.txt` for complete package dependencies

### Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/[username]/savelli-kustec-gal-sfh.git
   cd savelli-kustec-gal-sfh
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in development mode:

   ```bash
   pip install -e .
   ```

## Data

The repository includes data from two main sources:

### 1. Iyer et al. 2020 SFH Data

Star formation histories from multiple cosmological simulations:

- **EAGLE**, **Illustris**, **IllustrisTNG**, **Mufasa**, **Simba**, **SC-SAM**, **UniverseMachine**
- Located in `data/Iyer_etal_2020/`
- Original paper: [Iyer et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.498.430I)

### 2. Autoencoder Results  

Predictions from multimodal autoencoder analysis:

- SFH, SFR, and simulation predictions with/without weights
- Located in `data/autoencoder_results/`
- Generated using [Multimodal-Autoencoder](https://github.com/harrypenguin/Multimodal-Autoencoder)

## Usage

### Quick Start

```python
from src.utils.data_processing import load_iyer_data, prepare_umap_data
from src.utils.plotting import draw_grid_A
from src.utils.analysis import P_s_x

# Load simulation data
sim_data = load_iyer_data()

# Prepare data for UMAP analysis
combined_sfh, labels = prepare_umap_data(sim_data)

# Create plotting grid
axes = draw_grid_A(xlabel='UMAP 1', ylabel='UMAP 2')
```

### Analysis Notebooks

The analysis is organized into focused notebooks:

1. **01_data_preparation.ipynb** - Load and preprocess all data files
2. **02_main_sequence.ipynb** - Galaxy main sequence analysis and mass cuts  
3. **03_sfh_analysis.ipynb** - Normalized SFH analysis and sample plots
4. **04_umap_analysis.ipynb** - UMAP embedding and battleship grid analysis
5. **05_sop_analysis.ipynb** - Simulation of origin probabilities and Shannon entropy
6. **06_autoencoder.ipynb** - Autoencoder results and confusion matrices
7. **07_mwa_analysis.ipynb** - Milky Way analogue identification and analysis

### Key Analysis Features

#### UMAP Analysis

- Dimensionality reduction of normalized SFHs into 2D embedding space
- "Battleship grid" visualization (10×10 grid labeled A-J, 1-10)
- Grid cell averaging and statistical analysis

#### Simulation of Origin Probabilities (SOPs)

- Bayesian classification of galaxies by parent simulation
- Kernel density estimation in UMAP space
- Shannon entropy calculations for classification uncertainty

#### Milky Way Analogues

- Observationally Selected Analogues (OSAs) based on stellar mass and SFR
- Weighted quantile analysis through cosmic time
- Effective sample size tracking and fraction calculations

## Key Functions

### Analysis Functions (`src/utils/analysis.py`)

- `boxing()` - Grid-based data binning and averaging
- `P_s_x()` - Calculate simulation of origin probabilities  
- `calc_weights()` - Calculate OSA weights from χ² distributions
- `quantile()` - Weighted quantile calculations

### Plotting Functions (`src/utils/plotting.py`)

- `draw_battleship_grid()` - Draw 10×10 labeled grid
- `draw_grid_A/B/C/D()` - Various multi-panel plot layouts
- `plot_points()` - Scatter plots with colormapping
- `plot_averages()` - Grid cell averages with embedded SFH plots

### Data Processing (`src/utils/data_processing.py`)

- `load_iyer_data()` - Load simulation SFH data from .mat files
- `load_autoencoder_data()` - Load autoencoder predictions from .npy files  
- `apply_mass_cuts()` - Apply stellar mass cuts to simulations
- `prepare_umap_data()` - Prepare normalized SFH data for UMAP analysis

## Figures

The notebooks reproduce all figures from the paper:

- **Figure 1**: Galaxy main sequence plots for all simulations
- **Figure 2**: Sample normalized SFHs from each simulation  
- **Figure 3**: UMAP projection colored by simulation origin
- **Figure 4**: Kernel density estimation in UMAP space
- **Figure 5**: UMAP projections colored by stellar mass and SFR
- **Figure 6**: Simulation of origin probability maps
- **Figure 7**: Shannon entropy in UMAP space
- **Additional figures**: Autoencoder analysis and MWA results

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{savelli2024sfh,
  title={Classifying Galaxy Star Formation Histories with Machine Learning},
  author={Savelli, R. and Kustec, [First Name] and collaborators},
  journal={Journal Name},
  year={2024},
  note={in preparation}
}
```

Also cite the original data source:

```bibtex
@article{iyer2020cosmic,
  title={The Cosmic Baryon and Metal Cycles},
  author={Iyer, Kartheik G. and others},
  journal={MNRAS},
  volume={498},  
  pages={430},
  year={2020}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the analysis or code, please contact:

- [Primary Author Name] - [email]
- [Secondary Author Name] - [email]

## Acknowledgments

- Kartheik G. Iyer for providing the original SFH simulation data
- The cosmological simulation teams (EAGLE, Illustris, etc.) for the underlying data
- Contributors to the open-source packages used in this analysis
