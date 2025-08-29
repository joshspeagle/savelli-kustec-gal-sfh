#############################################################
#                                                           #
#           Data loading and processing utilities          #
#                                                           #
#############################################################

import numpy as np
import os
from pathlib import Path
from scipy.io import loadmat
from scipy.integrate import simpson

from .analysis import sim_name, times, zoom_name

# =============================================================================
# Data Loading Functions
# =============================================================================

def get_data_path():
    """
    Get the path to the data directory.
    
    Returns
    -------
    Path
        Path to data directory
    """
    return Path(__file__).parent.parent.parent / "data"

def load_iyer_data(sim_name_list=None, data_path=None):
    """
    Load Iyer et al. 2020 SFH data from .mat files.
    
    Parameters
    ----------
    sim_name_list : list, optional
        List of simulation names to load. If None, loads all available simulations.
    data_path : str or Path, optional
        Path to data directory. If None, uses default data path.
    
    Returns
    -------
    sim_data : dict
        Dictionary containing SFH data for each simulation
    """
    if data_path is None:
        data_path = get_data_path() / "Iyer_etal_2020"
    else:
        data_path = Path(data_path)
    
    if sim_name_list is None:
        sim_name_list = sim_name
    
    sim_data = {}
    
    for sim in sim_name_list:
        # Construct filename - handle case differences in file names
        if sim == 'EAGLE':
            filename = "Eagle_sfhs_psds.mat"
        elif sim == 'SC-SAM':
            filename = f"{sim}_sfhs_psds.mat"
        else:
            filename = f"{sim}_sfhs_psds.mat"
        
        filepath = data_path / filename
        
        if filepath.exists():
            try:
                mat_data = loadmat(filepath)
                # Extract data based on the actual mat file structure (match original exactly)
                sim_data[sim] = {
                    'sfh_raw': mat_data.get('smallsfhs', None).T if mat_data.get('smallsfhs', None) is not None else None,  # Transpose like original: (galaxies, time)
                    'times': mat_data.get('smalltime', None),    # Time array
                    'sm': 10**mat_data.get('logmass', np.array([[]]))[0],  # Stellar mass (flatten)
                    'ngal': mat_data['logmass'].shape[1] if 'logmass' in mat_data else 0,  # Number of galaxies
                }
                
                print(f"Loaded {sim} data: {sim_data[sim]['ngal']} galaxies")
            except Exception as e:
                print(f"Error loading {sim}: {e}")
                sim_data[sim] = None
        else:
            print(f"File not found for {sim}: {filepath}")
            sim_data[sim] = None
    
    return sim_data

def load_autoencoder_data(data_path=None):
    """
    Load autoencoder prediction results from .npy files.
    
    Parameters
    ----------
    data_path : str or Path, optional
        Path to autoencoder data directory. If None, uses default data path.
    
    Returns
    -------
    ae_data : dict
        Dictionary containing autoencoder predictions
    """
    if data_path is None:
        data_path = get_data_path() / "autoencoder_results"
    else:
        data_path = Path(data_path)
    
    ae_data = {}
    
    # List of expected files
    expected_files = [
        'predictions_sfh.npy',
        'predictions_sfh_w.npy',
        'predictions_sfr.npy', 
        'predictions_sfr_w.npy',
        'predictions_sim.npy',
        'predictions_sim_w.npy'
    ]
    
    for filename in expected_files:
        filepath = data_path / filename
        if filepath.exists():
            try:
                data = np.load(filepath)
                # Remove file extension and use as key
                key = filename.replace('.npy', '')
                ae_data[key] = data
                print(f"Loaded {filename}: shape {data.shape}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"File not found: {filepath}")
    
    return ae_data

def load_zoom_data(data_path=None):
    """
    Load zoom simulation data from .mat files.
    
    Parameters
    ----------
    data_path : str or Path, optional
        Path to data directory. If None, uses default data path.
    
    Returns
    -------
    zoom_data : dict
        Dictionary containing zoom SFH data for each simulation
    """
    if data_path is None:
        data_path = get_data_path() / "Iyer_etal_2020"
    else:
        data_path = Path(data_path)
    
    zoom_data = {}
    
    for zoom in zoom_name:
        # Handle filename mapping
        if zoom == 'MarvelJL':
            filename = "Marvel_JL_sfhs_psds.mat"
        else:
            filename = f"{zoom}_sfhs_psds.mat"
        
        filepath = data_path / filename
        
        if filepath.exists():
            try:
                mat_data = loadmat(filepath)
                # Extract data based on the actual mat file structure (match original exactly)
                zoom_data[zoom] = {
                    'sfh_raw': mat_data.get('smallsfhs', None).T if mat_data.get('smallsfhs', None) is not None else None,  # Transpose like original: (galaxies, time)
                    'times': mat_data.get('smalltime', None),    # Time array
                    'sm': 10**mat_data.get('logmass', np.array([[]]))[0],  # Stellar mass (flatten)
                    'ngal': mat_data['logmass'].shape[1] if 'logmass' in mat_data else 0,  # Number of galaxies
                }
                
                print(f"Loaded {zoom} data: {zoom_data[zoom]['ngal']} galaxies")
            except Exception as e:
                print(f"Error loading {zoom}: {e}")
                zoom_data[zoom] = None
        else:
            print(f"File not found for {zoom}: {filepath}")
            zoom_data[zoom] = None
    
    return zoom_data

# =============================================================================
# Data Processing Functions  
# =============================================================================

def normalize_sfh(sfh_data, method='integrated'):
    """
    Normalize SFH data.
    
    Parameters
    ----------
    sfh_data : array_like
        Star formation history data
    method : str, default='integrated'
        Normalization method ('integrated', 'peak', 'none')
    
    Returns
    -------
    array_like
        Normalized SFH data
    """
    if method == 'integrated':
        # Normalize by integrated SFH
        integrated_sfh = np.trapz(sfh_data, dx=0.1, axis=1)
        return sfh_data / integrated_sfh[:, np.newaxis]
    elif method == 'peak':
        # Normalize by peak SFR
        peak_sfr = np.max(sfh_data, axis=1)
        return sfh_data / peak_sfr[:, np.newaxis]
    elif method == 'none':
        return sfh_data
    else:
        raise ValueError("Method must be 'integrated', 'peak', or 'none'")

def apply_mass_cuts(sim_data, cuts=None):
    """
    Apply mass cuts to simulation data.
    
    Parameters
    ----------
    sim_data : dict
        Simulation data dictionary
    cuts : dict, optional
        Dictionary of mass cuts for each simulation.
        Default cuts: 10^10 for Mufasa, Simba; 10^9 for everything else
    
    Returns
    -------
    dict
        Filtered simulation data
    """
    if cuts is None:
        # Default mass cuts
        cuts = {
            'EAGLE': 1e9,
            'Illustris': 1e9,
            'IllustrisTNG': 1e9,
            'Mufasa': 1e10,
            'Simba': 1e10,
            'SC-SAM': 1e9,
            'UniverseMachine': 1e9
        }
    
    filtered_data = {}
    
    for sim, data in sim_data.items():
        if data is None:
            filtered_data[sim] = None
            continue
            
        if sim in cuts and 'sm' in data and data['sm'] is not None:
            mask = data['sm'] >= cuts[sim]
            
            filtered_data[sim] = {}
            for key, values in data.items():
                if key == 'sm' and values is not None:
                    # Handle stellar mass (1D array)
                    filtered_data[sim][key] = values[mask]
                elif key == 'sfh_raw' and values is not None:
                    # Handle SFH data (2D array: galaxies x time, like original)
                    filtered_data[sim][key] = values[mask, :]
                elif key == 'ngal':
                    # Update galaxy count
                    filtered_data[sim][key] = np.sum(mask)
                else:
                    # Keep scalar values or unchanged arrays
                    filtered_data[sim][key] = values
                    
            print(f"{sim}: {np.sum(mask)}/{len(mask)} galaxies after M* > {cuts[sim]:.0e} cut")
        else:
            filtered_data[sim] = data
    
    return filtered_data

def remove_zero_sfhs(sim_data):
    """
    Remove galaxies with all-zero star formation histories.
    
    Parameters
    ----------
    sim_data : dict
        Simulation data dictionary
    
    Returns
    -------
    dict
        Filtered simulation data
    """
    filtered_data = {}
    
    for sim, data in sim_data.items():
        if data is None or 'sfh_raw' not in data or data['sfh_raw'] is None:
            filtered_data[sim] = data
            continue
            
        # Find non-zero SFHs (sfh_raw is galaxies x time, like original)
        sfh_raw = data['sfh_raw']
        non_zero_mask = np.any(sfh_raw > 0, axis=1)  # Check along time axis (axis=1 now)
        
        filtered_data[sim] = {}
        for key, values in data.items():
            if key == 'sm' and values is not None:
                # Handle stellar mass (1D array)
                filtered_data[sim][key] = values[non_zero_mask]
            elif key == 'sfh_raw' and values is not None:
                # Handle SFH data (2D array: galaxies x time, like original)
                filtered_data[sim][key] = values[non_zero_mask, :]
            elif key == 'ngal':
                # Update galaxy count
                filtered_data[sim][key] = np.sum(non_zero_mask)
            else:
                # Keep scalar values or unchanged arrays
                filtered_data[sim][key] = values
                
        print(f"{sim}: {np.sum(non_zero_mask)}/{len(non_zero_mask)} galaxies after removing zero SFHs")
    
    return filtered_data

def interpolate_to_common_grid(sfh_data, input_times=None, output_times=None):
    """
    Interpolate SFH data to a common time grid.
    
    Parameters
    ----------
    sfh_data : array_like
        Star formation history data
    input_times : array_like, optional
        Input time grid. If None, assumes default grid.
    output_times : array_like, optional
        Output time grid. If None, uses default times array.
    
    Returns
    -------
    array_like
        Interpolated SFH data
    """
    if output_times is None:
        output_times = times
        
    if input_times is None:
        # Assume input matches output grid
        return sfh_data
    
    # Perform interpolation for each galaxy
    interpolated_sfh = np.zeros((sfh_data.shape[0], len(output_times)))
    
    for i in range(sfh_data.shape[0]):
        interpolated_sfh[i] = np.interp(output_times, input_times, sfh_data[i])
    
    return interpolated_sfh

def prepare_umap_data(sim_data, normalize=True, remove_zeros=True):
    """
    Prepare SFH data for UMAP analysis. This function processes the data following
    the original notebook workflow: normalize by integrated SFH, then interpolate
    to common time grid.
    
    Parameters
    ----------
    sim_data : dict
        Simulation data dictionary
    normalize : bool, default=True
        Whether to normalize SFHs by integrated value
    remove_zeros : bool, default=True
        Whether to remove zero SFHs
    
    Returns
    -------
    combined_sfh : array_like
        Combined SFH array for UMAP (galaxies x time_bins)
    combined_labels : array_like
        Simulation labels for each galaxy
    """
    # Common time grid: 0.1 to 13.6 Gyr in 0.1 Gyr steps
    common_times = np.arange(0.1, 13.7, 0.1)
    
    all_sfhs = []
    all_labels = []
    
    for sim_idx, (sim, data) in enumerate(sim_data.items()):
        if data is None or 'sfh_raw' not in data or data['sfh_raw'] is None:
            continue
            
        # Get the time array and SFH data
        sim_times = data['times'][0] if data['times'] is not None else times
        sfh_raw = data['sfh_raw']  # (galaxies x time) - matches original
        
        # Process each galaxy
        processed_sfhs = []
        
        for i in range(sfh_raw.shape[0]):  # Loop over galaxies (first dimension now)
            sfh_galaxy = sfh_raw[i, :]  # Get time series for galaxy i
            
            # Skip if remove_zeros and all zeros
            if remove_zeros and np.all(sfh_galaxy == 0):
                continue
            
            # Normalize by integrated SFH if requested
            if normalize:
                integral = simpson(sfh_galaxy, x=sim_times * 1e9)  # Convert to years
                if integral > 0:
                    sfh_galaxy = sfh_galaxy / integral
                else:
                    continue  # Skip galaxies with zero integral
            
            # Interpolate to common time grid
            sfh_interp = np.interp(common_times, sim_times, sfh_galaxy)
            processed_sfhs.append(sfh_interp)
        
        if processed_sfhs:
            processed_sfhs = np.array(processed_sfhs)
            all_sfhs.append(processed_sfhs)
            all_labels.extend([sim_idx] * len(processed_sfhs))
    
    if not all_sfhs:
        return np.array([]), np.array([])
    
    combined_sfh = np.vstack(all_sfhs)
    combined_labels = np.array(all_labels)
    
    print(f"Combined data: {len(combined_sfh)} total galaxies with {combined_sfh.shape[1]} time bins")
    
    return combined_sfh, combined_labels

# =============================================================================
# Milky Way Values
# =============================================================================

def get_milky_way_values():
    """
    Get Milky Way observational constraints.
    
    Returns
    -------
    dict
        Dictionary of Milky Way properties
    """
    return {
        'stellar_mass': 5e10,  # Solar masses
        'stellar_mass_error': 1e10,
        'sfr': 1.65,  # Solar masses per year  
        'sfr_error': 0.19,
        'metallicity': 0.0,  # Solar metallicity
        'metallicity_error': 0.1
    }