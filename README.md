# Star Formation History Analysis

Repository containing the data and code used in **Savelli & Kustec et al.**.

## Overview

This repository provides a comprehensive analysis of star formation histories (SFHs) from multiple galaxy simulations using machine learning techniques (UMAP and autoencoder-driven comparisons).

## Repository Structure

```bash
savelli-kustec-gal-sfh/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── data/                            # Data directory
│   ├── Iyer_etal_2020_SFH_data/     # SFH data from Iyer et al. 2020
│   └── autoencoder_results/         # Autoencoder prediction results  
├── src/                             # Source code
│   └── utils/                       # Utility modules
└── notebooks/                       # Analysis notebooks
```

## Installation

You can "install" the repository to make sure you have all the dependencies needed to recompute the plots. Otherwise, you can just check what's available in the `requirements.txt` and/or `setup.py` files.

## Data

The repository includes data from two main sources:

### 1. Iyer et al. 2020 SFH Data

Star formation histories from multiple cosmological simulations:

- **EAGLE**, **Illustris**, **IllustrisTNG**, **Mufasa**, **Simba**, **SC-SAM**, **UniverseMachine**

While most data is available through the repo and managed with Git LFS, the original SFH data from Iyer et al. (2020) could not be uploaded due to the large file sizes. Those can instead be downloaded directly at [this link](https://www.dropbox.com/scl/fi/y40lkcyq547i99fpzidyv/Iyer_etal_2020_SFH_data.zip?rlkey=znhkushf5vhr01hrs07m98t1j&dl=0).

### 2. Autoencoder Results  

Predictions from multimodal autoencoder analysis:

- SFH, SFR, and simulation predictions with/without number count-based weights

These were generated using [this implementation](https://github.com/harrypenguin/Multimodal-Autoencoder) and are described in more detail in the main text.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the analysis or code, please contact:

- Alicia Savelli - <alicia.savelli@utoronto.ca>
- Josh Speagle - <j.speagle@utoronto.ca>
