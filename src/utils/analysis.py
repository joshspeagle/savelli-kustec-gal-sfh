#############################################################
#                                                           #
#           Analysis functions for SFH research            #
#                                                           #
#############################################################

import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import simpson
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import umap

# Constants and labels
SM_axis_label = r'$M_\star\ [\mathrm{M}_\odot]$'
SFR_axis_label = r'$\dot{M}_\star\ [\mathrm{M}_\odot\mathrm{yr}^{-1}]$'
sSFR_axis_label = r'$\dot{M}_\star/m_\star\ [\mathrm{yr}^{-1}]$'
t_axis_label = 'Time [Gyr]'

sim_name = np.array(['EAGLE',
                     'Illustris',
                     'IllustrisTNG',
                     'Mufasa',
                     'Simba',
                     'SC-SAM',
                     'UniverseMachine'])
sim_name_short = ['EAGLE', 'Illustris', 'TNG', 'Mufasa', 'Simba', 'SC-SAM', 'UM']

times = np.arange(0, 14.1, 0.1)[1:137]

# Color definitions for plotting
colors = {'EAGLE':'limegreen', 'Illustris':'deepskyblue','IllustrisTNG':'royalblue', 
          'Mufasa':'darkorange', 'Simba':'gold', 
          'SC-SAM':'hotpink', 'UniverseMachine':'darkorchid'}

zoom_name = np.array(['FIRE-2', 'g14', 'MarvelJL'])

zoom_colors = {'FIRE-2': 'r', 'g14': 'g', 'MarvelJL': 'b'}

zoom_markers = {'FIRE-2': '^', 'g14': 'P', 'MarvelJL': '*'}

# =============================================================================
# Utility Functions
# =============================================================================

def boxing(x, y, z, n=10, average=False, q=None, axis=0, ess=False):
    """
    Chunk data into battleship grid.
    
    Parameters
    ----------
    x : array_like
        x-coordinate of data
    y : array_like
        y-coordinate of data  
    z : array_like
        value of data
    n : int, default=10
        Grid size (n x n)
    average : bool, default=False
        Take average in the given cell
    q : list, optional
        Percentiles to calculate
    axis : int, default=0
        Axis along which to compute percentiles
    ess : bool, default=False
        Calculate effective sample size per cell
    
    Returns
    -------
    avg : ndarray
        Processed values for each grid cell
        
    Notes
    -----
    Do not use both average and q parameters simultaneously.
    """
    avg = []
    for x1 in range(n):
        for y1 in range(n):
            x2 = x1+1
            y2 = y1+1
            
            if (x2<n) & (y2<n): 
                box = z[(x >= x1) & (x < x2) & (y >= y1) & (y < y2)]
            elif (x2<n) & (y2==n):
                box = z[(x >= x1) & (x < x2) & (y >= y1) & (y <= y2)]
            elif (x2==n) & (y2<n):
                box = z[(x >= x1) & (x <= x2) & (y >= y1) & (y < y2)]
            elif (x2<n) & (y2==n):
                box = z[(x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)]
                
            if average:
                box = np.nanmean(box, axis=0)
                
            if q is not None:
                if box.size == 0:
                    box = np.full((len(q), z.shape[1]), np.nan)
                else:
                    box = np.percentile(box, q, axis=axis)
                
                new_shape = (n, n) + box.shape
            else:
                new_shape = (n, n)
                
            if ess:
                box = np.sum(box)**2/np.sum(box**2)
            
            avg.append(box)

    # Use object array to handle variable-length arrays
    if q is not None:
        avg = np.reshape(np.array(avg), new_shape)
    else:
        # Create object array for variable-length data
        result = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                result[i, j] = avg[i * n + j]
        avg = result
    
    return avg

def P_x_s(x, s, umaps):
    """
    Probability of position x given simulation s -- equivalent to calculating the KDE of simulations at position x.
    
    Parameters
    ----------
    x : array_like
        Position array
    s : array_like
        Simulation sSFHs umap embedding
    umaps : array_like
        Array of umap positions sorted into 10x10 battleship gridcells
    
    Returns
    -------
    p : ndarray
        Probabilities
    """
    kde   = gaussian_kde(np.array([s[:,0], s[:,1]]))
    p_x_s = kde.evaluate(x).reshape((int(np.sqrt(x.shape[1])),int(np.sqrt(x.shape[1])))) # reshape to (100,100)
    
    # clean data
    p_x_s = p_x_s / np.sum(p_x_s) * 1e4 # normalize to integrate to 100x100
    
    for i in range(10):
        for j in range(10):
            if umaps[i,j].shape[0] < 100:
                p_x_s[i*10:i*10+10,j*10:j*10+10] = np.full((10,10),np.nan) # get rid of boxes with too few points
    
    return p_x_s
    
def P_s_x(x, sim_name, sim_data, umaps, normint=True, bias=False, mask=False):
    """
    Simulation of origin probabilities -- probability of simulation s given position x.
    
    Parameters
    ----------
    x : array_like
        Position array
    sim_name : list
        Simulation names
    sim_data : dict
        Simulation data set
    umaps : array_like
        Array of umap positions sorted into 10x10 battleship gridcells
    normint : bool, default=True
        If true, normalize the sum over simulations of KDEs to 10^4
        If false, normalize the sum of each individual KDE to 10^4
    bias : bool, default=False
        If true, use fraction of galaxies from a given simulation as p_s (biases towards sims with higher galaxy counts)
        If false, use equal 1/7 for all sims
    mask : bool, default=False
        SM cut at 10^10 MSun for all simulations
    
    Returns
    -------
    p : dict
        Probabilities
    """
    
    # Probability of coming from any one simulation is equal
    p_s = dict.fromkeys(sim_name)
    # Probability of position x given simulation s
    p_x_s = dict.fromkeys(sim_name) 
    
    for sim in sim_name:
        m = sim_data[sim]['sm']>=1e10
        if mask:
            n_gal = np.sum(m)
            N_gal = np.sum([np.sum(sim_data[sim]['sm']>=1e10) for sim in sim_name])
            
            s = sim_data[sim]['umap'][m]
        else:
            n_gal = sim_data[sim]['ngal']
            N_gal = np.sum([sim_data[sim]['ngal'] for sim in sim_name])
            
            s = sim_data[sim]['umap']
        
        # Probability of coming from a given simulation
        if bias:
            p_s[sim] = n_gal / N_gal 
        else:
            p_s[sim] = 1/7
            
        # Probability of position x given simulation s
        p_x_s[sim] = P_x_s(x, s, umaps)
        if normint:
            p_x_s[sim] *= n_gal/N_gal # normalize again?
    
    # Probability of position x (Sum over individual sim KDEs * p_s)
    p_x = np.sum(np.array([p_x_s[sim] * p_s[sim] for sim in sim_name]), axis=0) 
    
    # Simulation of origin probabilities
    p_s_x = dict.fromkeys(sim_name)
    for sim in sim_name:
        p_s_x[sim] = (p_x_s[sim] * p_s[sim]) / p_x
        
    return p_s_x

# =============================================================================
# MWA Analysis Functions
# =============================================================================

def calc_pdf(data1, data2=None):
    """ 
    Calculate pdf from 1 or 2 dimensional gaussian KDE.
    
    Parameters
    ----------
    data1 : array_like
        Input data.
    data2 : array_like, optional
        Input data for optional 2nd dimension.
        
    Returns
    -------
    pdf : array_like
        The calculated pdf.
    """
    
    # log data
    if data2 is None:
        logged_data = np.log10([data1])
    else:
        logged_data = np.log10(np.c_[data1,data2].T)
    # calculate pdf
    pdf = gaussian_kde(logged_data[:,np.all(np.isfinite(logged_data),axis=0)]).pdf(logged_data)
    pdf[pdf<=0] = np.nan # replace -inf with nan
    
    return pdf

def calc_weights(data1, mu1, sigma1, data2=None, mu2=None, sigma2=None, data3=None, mu3=None, sigma3=None, correct=False, pdf=None):
    """ 
    Calculate weights from 1 or 2 dimensional chi^2 distribution, with corrective pdf optional. 
    Weights will give distribution of data about an expected value (mu) with given uncertainty (sigma).
    
    Parameters
    ----------
    data1 : array_like
        Input data.
    mu1 : float
        Expected value.
    sigma1 : float
        Uncertainty.    
    data2 : array_like, optional
        Input data for optional 2nd dimension.
    mu2 : float, optional
        Expected value (2nd dimension).
    sigma2 : float, optional
        Uncertainty (2nd dimension). 
    data3 : array_like, optional
        Input data for optional 3rd dimension.
    mu3 : float, optional
        Expected value (3rd dimension).
    sigma3 : float, optional
        Uncertainty (3rd dimension).
    correct : bool, default=False
        Correct chi^2 distribution with pdf.
    pdf : array_like, optional
        Pre-calculated PDF for correction.
        
    Returns
    -------
    W : array_like
        The calculated weights.
    ess : float
        The effective sample size of the weighted distribution.
    """
    
    # chi^2 distribution
    if mu2 == None:
        chi2 = ((data1-mu1)/sigma1)**2
    elif mu3 == None:
        chi2 = ((data1-mu1)/sigma1)**2 + ((data2-mu2)/sigma2)**2
    else:
        chi2 = ((data1-mu1)/sigma1)**2 + ((data2-mu2)/sigma2)**2 + ((data3-mu3)/sigma3)**2
    W = np.exp(-chi2/2) # weights from chi^2
    if correct == True:
        # calculate pdf
        if pdf is None:
            pdf = calc_pdf(data1,data2)
        W /= pdf # adjust weights by dividing by pdf
    W[~np.isfinite(W)]=0 # get rid of -inf
    ess = np.sum(W)**2/np.sum(W**2) # effective sample size
    
    return W, ess

def weighted_std(values, weights):
    """
    Return the weighted standard deviation.

    Parameters
    ----------
    values : array_like
        Input values.
    weights : array_like
        Associated weights.
        
    Returns
    -------
    float
        Weighted standard deviation.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)

def quantile(x, q, weights=None):
    """
    Compute (weighted) quantiles from an input set of samples.
    
    Parameters
    ----------
    x : array_like
        Input samples.
    q : array_like
        The list of quantiles to compute from [0., 1.].
    weights : array_like, optional
        The associated weight from each sample.
        
    Returns
    -------
    quantiles : array_like
        The weighted sample quantiles computed at q.
    """
    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)
    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")
    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q))
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x).")
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()
        return quantiles
    
def quantile_hists(data, q=[0.16, 0.5, 0.84], weights=None):
    """ 
    Compute (weighted) quantiles from an input dataset over nsnaps snapshots.
    
    Parameters
    ----------
    data : array_like
        Input data.
    q : array_like, default=[0.16, 0.5, 0.84]
        The list of quantiles to compute from [0., 1.].
    weights : array_like, optional
        The associated weight from each sample.
        
    Returns
    -------
    quantiles : array_like
        The weighted sample quantiles computed at q over nsnaps snapshots.
    """
    
    nsnaps=len(data)
    
    qs = []
    
    for t in range(nsnaps):
        qs_t = quantile(data[t], q=q, weights=weights)
        qs.append(qs_t)
        
    return np.array(qs).T

def calc_weights_hists(data1, data2=None, weights=None):
    """ 
    Calculate OSA weights back through time over nsnaps snapshots.
    
    Parameters
    ----------
    data1 : array_like
        Input data with shape (nsnaps, ngal).
    data2 : array_like, optional
        Input data for optional second dimension with shape (nsnaps, ngal).
    weights : array_like, optional
        Redshift 0 weights to apply at each timestep. If None, computes weights based on regular quantiles.
        
    Returns
    -------
    Ws : array_like
        OSA weights at each snapshot.
    ess : array_like
        Effective sample sizes of OSAs at each snapshot.
    """
    
    # find number of snapshots to iterate over
    nsnaps = data1.shape[0]
    
    # empty lists to store weights and ess
    Ws = []
    ess = []
    
    for t in range(nsnaps):        
        # compute weighted quantiles at each time step
        qt_data1 = quantile(data1[t], q=[0.16,0.5,0.84], weights=weights)
        
        # compute new weights
        if data2 is None:
            W_t, ess_t = calc_weights(data1=data1[t], mu1=qt_data1[1], sigma1=(qt_data1[2]-qt_data1[0])/2) 
        else:
            qt_data2 = quantile(data2[t], q=[0.16,0.5,0.84], weights=weights)
            W_t, ess_t = calc_weights(data1=data1[t], mu1=qt_data1[1], sigma1=(qt_data1[2]-qt_data1[0])/2,
                                       data2=data2[t], mu2=qt_data2[1], sigma2=(qt_data2[2]-qt_data2[0])/2)
        
        # append to empty lists
        Ws.append(W_t)
        ess.append(ess_t)
        
    return np.array(Ws), np.array(ess)

def calc_fosa(ess_hist):
    """ 
    Calculate fraction of observationally selected analogues going back in time.
    
    Parameters
    ----------
    ess_hist : array_like
        Effective sample size of OSAs at each snapshot.
        
    Returns
    -------
    f : array_like
        The calculated fraction.
    """
    
    f = ess_hist[-1]/ess_hist
    
    return f

def calc_dsfr(SFR_hist, weights):
    """ 
    Calculate the SFR population dispersion in a given analogue group.
    
    Parameters
    ----------
    SFR_hist : array_like
        SFR history with shape (nsnaps, ngal).
    weights : array_like
        Redshift 0 weights.
        
    Returns
    -------
    dsfr : array_like
        The calculated SFR population dispersion.
    """
    
    # find nsnaps
    nsnaps = SFR_hist.shape[0]
    dSFR = np.zeros(nsnaps)
    
    # calculate dSFR at each snapshot
    for t in range(nsnaps):
        qt = quantile(SFR_hist[t], [0.16,0.84], weights = weights)
        dSFR[t] = (qt[1]-qt[0])/2
        
    return dSFR