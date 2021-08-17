import pandas as pd
import numpy as np
#from tqdm import tqdm
from scipy.stats import mannwhitneyu, fisher_exact, ttest_ind, ttest_ind_from_stats, norm, lognorm, t as tdist
from statsmodels.stats.power import tt_ind_solve_power
from statsmodels.api import OLS, WLS
from outliers import adjbox, logbox
from numba import jit
from numpy.random import default_rng, SeedSequence
from functools import partial
from scipy.optimize import root_scalar, minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
#from multiprocessing import Pool
#from os import cpu_count

from time import perf_counter


def select_strat(df, cut):
    '''
    Auxiliary function. Returns a boolean mask for a match in a multi-column segment
    '''
    cols = df.columns
    if len(cols) == 1:
        return df[cols[0]] == cut
    else:
        res = np.empty(df.shape, dtype= np.bool)
        for k in range(len(cols)):
            res[:,k] = df.iloc[:,k] == cut[k]
        return np.all(res, axis= 1)

def mannu_CI_shift(sample1, sample2, alpha= 0.05, n_grid= 25):
    '''
    (old version) Returns a confidence interval for a mean difference based on the Mann-Whitney U test. It is based on the distribution of differences between two samples (sample1 - sample2). This version should only be used with continuous distributions.
    '''
    n1 = len(sample1)
    n2 = len(sample2)
    n_diffs = n1 * n2
    
    mannu_result = mannwhitneyu(sample1, sample2, alternative= 'two-sided')
    pv = mannu_result[1]
    
    Z = norm.ppf(1 - alpha/2)
    sd_stat = abs(mannu_result[0] - n_diffs/2) / norm.ppf(1 - pv/2)
    #k = np.ceil(n_diffs/2 - (Z * np.sqrt(n_diffs*(n1+n2+1)/12)))
    k = np.ceil(n_diffs/2 - (Z * sd_stat))
    
    auc_lo = (mannu_result[0] - (Z * sd_stat)) / n_diffs
    auc_up = (mannu_result[0] + (Z * sd_stat)) / n_diffs
    
    #If there are too many zeros in both samples, the confidence interval will be zeros
    if k > n_diffs - (sample1 == 0).sum() * (sample2 == 0).sum():
        return (0.0, 0.0, pv, auc_lo, auc_up)
    
    #If the difference matrix would have more than 400M elements (3.2 GB of float64),
    #a sample is taken to estimate an approximation of the CI,
    #then it is refined by computing the p-values of the u-test over a grid-search
    
    two_stage = n_diffs > 400000000
    if two_stage:
        
        if n1 > 10000:
            m1 = np.random.choice(sample1, size= 10000, replace= False)
        else:
            m1 = sample1
        if n2 > 10000:
            m2 = np.random.choice(sample2, size= 10000, replace= False)
        else:
            m2 = sample2

        diffs = (m1.reshape(-1,1) - m2.reshape(1,-1)).reshape(1,-1).squeeze()

        ci_pre = np.quantile(diffs, [(k-1)/n_diffs, (n_diffs-k+1)/n_diffs])        
        width = 5*(ci_pre[1] - ci_pre[0]) #ancho de grilla: 5 veces la estimación previa con muestra
        del diffs, m1, m2
        if width == 0.0:
            return (0.0, 0.0, pv, auc_lo, auc_up)
    
        center = ci_pre.mean()
        grid = np.linspace(center - width/2, center + width/2, n_grid)
        pvalues = np.empty_like(grid)

        for k in range(n_grid):
            pvalues[k] = mannwhitneyu(sample1, sample2 + grid[k], alternative= 'two-sided')[1]
        int_points = grid[pvalues > alpha]
        if len(int_points) > 0:
            return (int_points[0], int_points[-1], pv, auc_lo, auc_up)
        else:
            return (0.0, 0.0, pv, auc_lo, auc_up)
    else:
        diffs = (sample1.reshape(-1,1) - sample2.reshape(1,-1)).reshape(1,-1).squeeze()
        
        ci = np.quantile(diffs, [(k-1)/n_diffs, (n_diffs-k+1)/n_diffs])

        return (ci[0], ci[1], pv, auc_lo, auc_up)

@jit(nopython= True)
def mannu_grid_gen(sample2, replacer_z2, replacer_nz2, sampler2, loc, n2, n_zeros2, mu2):
    '''
    Auxiliary function for the Mann-Whitney U confidence interval.
    '''
    new2 = np.copy(sample2)
    loc = max(loc, 0.0)
    if loc >= 1.0: #Some zeros are replaced with non-zeros
        num_replace = min(int(np.round((np.sqrt(loc) - 1) * (n2 - n_zeros2))), n_zeros2)
        selec_idx = np.linspace(0.0,1.0,num_replace) if num_replace != 1 else np.array([0.5])
        aux_round = np.empty_like(selec_idx)
        np.round(selec_idx * (n2 - n_zeros2 - 2e-9) - 0.5 + 1e-9, 0, aux_round)
        new2[replacer_z2[:num_replace]] = sampler2[aux_round.astype(np.int64)]
    else: #Some non-zeros are replaced with zeros
        num_replace = min(int(np.round((1 - np.sqrt(loc)) * (n2 - n_zeros2))), n2 - n_zeros2)
        selec_idx = np.linspace(0.0,1.0,num_replace) if num_replace != 1 else np.array([0.5])
        aux_round = np.empty_like(selec_idx)
        np.round(selec_idx * (n2 - n_zeros2 - 2e-9) - 0.5 + 1e-9, 0, aux_round)
        new2[replacer_nz2[aux_round.astype(np.int64)]] = 0.0
    new2 *= (mu2 * loc / new2.mean()) if (not (new2 == 0).all()) else 1.0
    return new2
    
def mannu_prop_eval(data, treat_col, tgt_col, alpha= 0.05):
    '''
    Returns a confidence interval for a mean difference based on the Mann-Whitney U test. It is designed for data with a positive probability of being zero, and a continuous distribution for every other value.
    Perturbations are applied to the sample2 via a grid search, and p-values for the U-test are computed for each perturbation. The confidence interval consists of the points with p-value > alpha. The perturbations are multiplications of the data by a factor, and modification of the number of non-zeros (both effects yield a total shift in the mean). The position of the zeros is not changed.
    '''
    sample1 = np.sort(data.loc[data[treat_col] == 1, tgt_col].to_numpy(dtype= np.float32))
    sample2 = np.sort(data.loc[data[treat_col] == 0, tgt_col].to_numpy(dtype= np.float32))
    mannu_result = mannwhitneyu(sample1, sample2, alternative= 'two-sided')
    pv = mannu_result[1]
    
    Z = norm.ppf(1 - alpha/2)
    n1 = len(sample1)
    n2 = len(sample2)
    n_diffs = n1 * n2
    sd_stat = abs(mannu_result[0] - n_diffs/2) / norm.ppf(1 - pv/2)
    auc_lo = (mannu_result[0] - (Z * sd_stat)) / n_diffs
    auc_up = (mannu_result[0] + (Z * sd_stat)) / n_diffs
    
    quants1 = np.quantile(sample1, [alpha/2, 0.5, 1-alpha/2])
    quants2 = np.quantile(sample2, [alpha/2, 0.5, 1-alpha/2])
    half_ci_guess = 0.5 * np.sqrt((quants1[2] - quants1[0])**2 + (quants2[2] - quants2[0])**2) * np.sqrt(1/n1 + 1/n2)
    if half_ci_guess == 0.0:
        half_ci_guess = Z * np.sqrt(1/n1 + 1/n2) * np.sqrt((np.var(sample1, ddof= 1) + np.var(sample2, ddof= 1))/2)
        if half_ci_guess == 0.0:
            return (np.nan, np.nan, np.nan, pv, auc_lo, auc_up)

    mu2 = sample2.mean()
    center_guess = (quants1[1] - quants2[1]) / mu2 + 1
    step = half_ci_guess / mu2
    
    zeros2 = sample2 == 0.0
    n_zeros2 = np.sum(zeros2)
    sampler2 = sample2[~zeros2]
    replacer_z2 = np.arange(n2)[zeros2]
    replacer_nz2 = np.arange(n2)[~zeros2]
    
    #Find the incremental by maximizing the p-value
    
    func_opt = lambda x : - mannwhitneyu(sample1,
                                         mannu_grid_gen(sample2, replacer_z2, replacer_nz2, sampler2, x, n2, n_zeros2, mu2),
                                         alternative= 'two-sided')[1]
    
    opt_res = minimize_scalar(func_opt, bracket= (center_guess if center_guess != 1.0 else 1.05, 1.0), 
                              method= 'brent', options= {'xtol': 0.002, 'maxiter': 10})
    inc = opt_res.x
    if opt_res.fun > -alpha:
        return (np.nan, np.nan, np.nan, pv, auc_lo, auc_up)
    
    #Find one exterior point over the interval and one under the interval
    int_up, int_lo = inc, inc
    pv_int_up, pv_int_lo = -opt_res.fun, -opt_res.fun
    found_up = False
    ext_up = np.inf
    pv_ext_up = 0.0
    found_lo = False
    ext_lo = -np.inf
    pv_ext_lo = 0.0
    k = 0
    
    while not (found_up and found_lo):
        k_even = k % 2 == 0
        if (k_even and found_up) or (not k_even and found_lo):
            k += 1
            continue
        loc = max(inc + (1 if k_even else -1) * step * int((k+2)/2) * 2**int(k/4), 0.0)
        new2 = mannu_grid_gen(sample2, replacer_z2, replacer_nz2, sampler2, loc, n2, n_zeros2, mu2)
        loc_res = mannwhitneyu(sample1, new2, alternative= 'two-sided')
        
        is_auc_up = (loc_res[0] / n1 / n2) >= 0.5
        if loc_res[1] >= alpha:
            if is_auc_up:
                pv_int_lo = loc_res[1] if int_lo > loc else pv_int_lo
                int_lo = min(int_lo, loc)
            else:
                pv_int_up = loc_res[1] if int_up < loc else pv_int_up
                int_up = max(int_up, loc)
        else:
            if is_auc_up:
                found_lo = True
                pv_ext_lo = loc_res[1] if ext_lo < loc else pv_ext_lo
                ext_lo = max(ext_lo, loc)
            else:
                found_up = True
                pv_ext_up = loc_res[1] if ext_up > loc else pv_ext_up
                ext_up = min(ext_up, loc)
        k += 1
        if k >= 100:
            return (np.nan, (inc - 1) * mu2, np.nan, pv, auc_lo, auc_up)
    
    #Find the interval boundaries using a root solver
    
    func_root = lambda x : mannwhitneyu(sample1,
                                        mannu_grid_gen(sample2, replacer_z2, replacer_nz2, sampler2, x, n2, n_zeros2, mu2),
                                        alternative= 'two-sided')[1] - alpha
    
    if (pv_ext_lo <= alpha) and (pv_int_lo >= alpha) and (ext_lo != int_lo):
        ci_lo = root_scalar(func_root, method= 'brentq', bracket= (ext_lo,int_lo), xtol= 0.002, maxiter= 10).root
    elif ext_lo == int_lo:
        ci_lo = ext_lo
    else:
        ci_lo = np.nan
    if (pv_ext_up <= alpha) and (pv_int_up >= alpha) and (ext_up != int_up):
        ci_up = root_scalar(func_root, method= 'brentq', bracket= (ext_up,int_up), xtol= 0.002, maxiter= 10).root
    elif ext_up == int_up:
        ci_up = ext_up
    else:
        ci_up = np.nan
        
    return ((ci_lo - 1) * mu2, (inc - 1) * mu2, (ci_up - 1) * mu2, pv, auc_lo, auc_up)
       
def boot_mean(x, iters= 1000, iter_size= None, seed= None):
    '''
    Returns a bootstrap array of means of length iters, sampling iter_size elements for each mean from the array x. Sampling and averaging are executed in blocks with a target of 3.2 GB maximum memory (400M elements)
    '''
    rng = default_rng(seed)
    if iter_size is None:
        iter_size = len(x.reshape(-1))
    blocks = int(np.ceil(iter_size * iters / 400000000)) #Number of blocks
    b_size = int(np.ceil(iters / blocks)) #Number of iterations per block
    last_block = iters % b_size if iters % b_size != 0 else b_size #Number of iterations in the last block
    res = np.empty(iters, dtype= np.float64)
    for k in range(blocks-1):
        res[int(k*b_size) : int((k+1)*b_size)] = rng.choice(x, size= (b_size, iter_size), replace= True).mean(axis= 1)
    res[-last_block:] = rng.choice(x, size= (last_block, iter_size), replace= True).mean(axis= 1)
    return res

def boot_eval(data, treat_col, tgt_col, alpha= 0.05, iters= 1000, iter_size= None, seed= None, return_samples= False):
    '''
    Bootstrap confidence interval for the mean difference between two samples.
    '''
    new_seeds = seed.spawn(2)
    x1 = data.loc[data[treat_col] == 1, tgt_col].to_numpy()
    means1 = boot_mean(x1, iters= iters, iter_size= iter_size, seed= new_seeds[0])
    x0 = data.loc[data[treat_col] == 0, tgt_col].to_numpy()
    means0 = boot_mean(x0, iters= iters, iter_size= iter_size, seed= new_seeds[1])
    
    diffs = means1 - means0
    inc = x1.mean() - x0.mean()
    inc_ci = np.quantile(diffs, [alpha/2, 1-alpha/2])
    pv = ((diffs <= (inc - abs(inc))) | (diffs >= (inc + abs(inc)))).mean()
      
    if return_samples:
        return inc_ci[0], inc, inc_ci[1], pv, means1, means0
    else:
        return inc_ci[0], inc, inc_ci[1], pv

def bayes_smoothing_eval(data, treat_col, tgt_col, alpha= 0.05, iters= 100000, 
                         prior_strength= (0.1, 0.05), seed= None, return_samples= False):
    '''
    Bayesian evaluation of the mean difference between two samples, with Bernoulli-Lognormal likelihood and conjugate priors (Beta for Bernoulli probability, Normal and Inverse-Gamma for mu and sigma). Priors are based on pooled data and are the same for treatment and control, so the evaluation has a smoothing effect (incrementals are reduced and confidence intervals and smaller).
    The prior sample size is computed separately for the Beta and Lognormal components. The prior strength is controlled via a tuple of parameters. In each case the sample size is: prior_strength[0] * (number of observations required to measure a percent incremental of prior_strength[1] using a t-test, for the observed pooled mean and standard deviation). prior_strength[0] can be interpreted as the bias in the incremental estimation when the real incremental is prior_strength[1] and number of observations is exactly what is needed to measure it.
    '''
    has_zero = (data[tgt_col] == 0).any()
    has_cont = not np.array_equal(data[tgt_col], data[tgt_col].astype(bool))
    
    #Split the data into binary and continuous parts
    bin_gt = data.loc[data[treat_col] == 1, tgt_col] != 0.0
    bin_gc = data.loc[data[treat_col] == 0, tgt_col] != 0.0
    if has_cont:
        cont_gt = data.loc[data[treat_col] == 1, tgt_col][bin_gt]
        cont_gc = data.loc[data[treat_col] == 0, tgt_col][bin_gc]
    
    gtgc_ratio = 1.0 / (len(data) / data[treat_col].sum() - 1.0)
    if has_zero:
        #Beta prior parametrization (binary part)
        mean_bin = (data[tgt_col] != 0.0).mean()
        n_prior_bin = prior_strength[0] * tt_ind_solve_power(effect_size= prior_strength[1]*np.sqrt(mean_bin/(1-mean_bin)),
                                                     alpha= alpha, power= 0.8, ratio= gtgc_ratio, alternative= 'two-sided')

        Beta_a = mean_bin * n_prior_bin
        Beta_b = n_prior_bin - Beta_a
    
        #Beta posteriors computation
        Beta_a_gt = Beta_a + bin_gt.sum()
        Beta_b_gt = Beta_b + len(bin_gt) - (Beta_a_gt - Beta_a)
        Beta_a_gc = Beta_a + bin_gc.sum()
        Beta_b_gc = Beta_b + len(bin_gc) - (Beta_a_gc - Beta_a)
    
    if has_cont:
        #Lognormal prior parametrization (continuous part)
        cont_pooled = np.concatenate([cont_gt, cont_gc])

        if (cont_gc > 0).all() and (cont_gt > 0).all():
            shift = 0.0
            cont_gt_clean = cont_gt
            cont_gc_clean = cont_gc
        else:
            lo_cut, _ = adjbox(cont_pooled, mult= 2.0)
            cont_pooled_clean = cont_pooled[cont_pooled > lo_cut]
            _, shift, _ = lognorm.fit(cont_pooled_clean)
            shift = min(shift, 0.0)
            lo_cut2, _ = logbox(cont_pooled_clean - shift)
            lo_cut = max(lo_cut, lo_cut2)
            cont_gt_clean = cont_gt[cont_gt > lo_cut]
            cont_gc_clean = cont_gc[cont_gc > lo_cut]
        log_cont_gt = np.log(cont_gt_clean - shift)
        log_cont_gc = np.log(cont_gc_clean - shift)
        log_cont = np.concatenate([log_cont_gt, log_cont_gc])
        m_prior = log_cont.mean()
        s_prior = np.var(log_cont, ddof= 1)
        
        efs = prior_strength[1] * np.mean(np.exp(log_cont)) / np.std(np.exp(log_cont), ddof= 1)
        n_prior_cont = prior_strength[0] * tt_ind_solve_power(effect_size= efs, alpha= alpha, power= 0.8, 
                                                          ratio= gtgc_ratio, alternative= 'two-sided')

        #Lognormal posteriors computation
        n_post_gt = n_prior_cont + len(log_cont_gt)
        n_post_gc = n_prior_cont + len(log_cont_gc)
        m_gt = (m_prior * n_prior_cont + log_cont_gt.sum()) / n_post_gt
        m_gc = (m_prior * n_prior_cont + log_cont_gc.sum()) / n_post_gc
        s_post_gt = (s_prior * n_prior_cont + \
                     np.var(log_cont_gt) * len(log_cont_gt) + \
                     n_prior_cont * len(log_cont_gt) / n_post_gt * (log_cont_gt.mean() - m_prior)**2
                    ) / n_post_gt
        s_post_gc = (s_prior * n_prior_cont + \
                     np.var(log_cont_gc) * len(log_cont_gc) + \
                     n_prior_cont * len(log_cont_gc) / n_post_gc * (log_cont_gc.mean() - m_prior)**2
                    ) / n_post_gc
    
    #Compute the incremental CI by Monte Carlo simulation of the expected value conditional on the parameters
    rng = default_rng(seed)
    if has_zero:
        probs_gt = rng.beta(Beta_a_gt, Beta_b_gt, iters)
        probs_gc = rng.beta(Beta_a_gc, Beta_b_gc, iters)
    else:
        probs_gt = np.ones(iters)
        probs_gc = np.ones(iters)
        n_prior_bin, Beta_a_gt, Beta_b_gt, Beta_a_gc, Beta_b_gc = np.full(5, np.nan)
    if has_cont:
        sigmas2_gt = 1.0 / rng.gamma(n_post_gt/2, 2/n_post_gt/s_post_gt, iters)
        sigmas2_gc = 1.0 / rng.gamma(n_post_gc/2, 2/n_post_gc/s_post_gc, iters)
        mus_gt = rng.normal(m_gt, np.sqrt(sigmas2_gt / n_post_gt))
        mus_gc = rng.normal(m_gc, np.sqrt(sigmas2_gc / n_post_gc))

        mean_post_gt = probs_gt * (shift + np.exp(mus_gt + sigmas2_gt / 2.0))
        mean_post_gc = probs_gc * (shift + np.exp(mus_gc + sigmas2_gc / 2.0))
        del probs_gt, probs_gc, mus_gt, mus_gc, sigmas2_gt, sigmas2_gc
    else:
        mean_post_gt = probs_gt
        mean_post_gc = probs_gc
        n_prior_cont, n_post_gt, n_post_gc, m_gt, m_gc, s_post_gt, s_post_gc, shift = np.full(8, np.nan)
    mean_diff = mean_post_gt - mean_post_gc
    
    inc = mean_diff.mean()
    inc_ci = np.quantile(mean_diff, [alpha/2, 1-alpha/2])
    pv = ((mean_diff <= (inc - abs(inc))) | (mean_diff >= (inc + abs(inc)))).mean()
    
    if return_samples:
        return (inc_ci[0], inc, inc_ci[1], pv, #Incrementals and p-value
                (n_prior_bin, Beta_a_gt, Beta_b_gt, Beta_a_gc, Beta_b_gc), #Beta parameters
                (n_prior_cont, n_post_gt, n_post_gc, m_gt, m_gc, s_post_gt, s_post_gc, shift), #Logn parameters
                mean_post_gt, mean_post_gc) #GT/GC samples (for stratified aggregation)
    else:
        return (inc_ci[0], inc, inc_ci[1], pv,
                (n_prior_bin, Beta_a_gt, Beta_b_gt, Beta_a_gc, Beta_b_gc), 
                (n_prior_cont, n_post_gt, n_post_gc, m_gt, m_gc, s_post_gt, s_post_gc, shift))

def e1_replacer(arr, e1_float):
    '''
    Auxiliary function for the Fisher evaluation
    '''
    tot1 = arr[:,1].sum()
    new_e1 = np.int32(np.round(e1_float))
    arr[:,1] = [tot1-new_e1, new_e1]
    return arr

def e0_replacer(arr, e0_float):
    '''
    Auxiliary function for the Fisher evaluation
    '''
    tot0 = arr[:,0].sum()
    new_e0 = np.int32(np.round(e0_float))
    arr[:,0] = [tot0-new_e0, new_e0]
    return arr

def fisher_eval(data, treat_col, tgt_col, alpha= 0.05):
    '''
    Returns a confidence interval for a mean difference based on a Fisher exact test (only for binary outcome). The points inside the interval are all the mean differences that, when added to the observed control mean, will not reject the null hypothesis for that modified data.
    '''
    
    #Compute the contingency table
    cont_table = data.dropna().groupby(treat_col)[tgt_col].value_counts().unstack(treat_col).fillna(0).astype(np.int32)[[0,1]].sort_index()
    if len(cont_table) < 2:
        return (0.0, 0.0, 0.0, np.nan)
    
    #P-value
    pv = fisher_exact(cont_table, alternative= 'two-sided')[1]
    
    #General parameters
    n1 = cont_table.loc[:,1].sum()
    mu1 = cont_table.loc[1,1] / n1
    n0 = cont_table.loc[:,0].sum()
    mu0 = cont_table.loc[1,0] / n0
    var1 = mu1*(1-mu1)
    var0 = mu0*(1-mu0)
    sterr = np.sqrt(var1 / n1 + var0 / n0)
    half_guess = norm.ppf(1 - alpha/2) * sterr
    inc = mu1 - mu0
    
    #Find the p-values at the extremes (the alpha for rejection cannot be lower)
    min_lo_alpha = fisher_exact(e0_replacer(cont_table.to_numpy().copy(), 0), alternative= 'two-sided')[1]
    min_up_alpha = fisher_exact(e0_replacer(cont_table.to_numpy().copy(), n0), alternative= 'two-sided')[1]
    
    #Find the interval boundaries using a root solver
    func_root = lambda x : fisher_exact(e0_replacer(cont_table.to_numpy().copy(), x),
                                        alternative= 'two-sided')[1] - alpha
    
    xtol = 0.002 * n0
    n_center = mu1 * n0
    if min_lo_alpha < alpha:
        n_guess = max(0, (mu1 - half_guess*0.9) * n0)
        n_guess2 = max(0, (mu1 - half_guess*1.1) * n0)
        root_res = root_scalar(func_root, method= 'brentq', bracket= (n_center, 0)
                               , x0= n_guess, x1= n_guess2, maxiter= 10, xtol= xtol)
        ci_lo = root_res.root
    else:
        ci_lo = 0
    
    if min_up_alpha < alpha:
        n_guess = min(n0, (mu1 + half_guess*0.9) * n0)
        n_guess2 = min(n0, (mu1 + half_guess*1.1) * n0)
        root_res = root_scalar(func_root, method= 'brentq', bracket= (n_center, n0)
                               , x0= n_guess, x1= n_guess2, maxiter= 10, xtol= xtol)
        ci_up = root_res.root
    else:
        ci_up = n0
        
    return (ci_lo/n0 - mu0, inc, ci_up/n0 - mu0, pv)
      
def outlier_lims(x, tgt_type):
    if tgt_type == 'binary':
        return (-np.inf, np.inf)
    elif tgt_type == 'non-neg':
        return (-np.inf, logbox(x)[1])
    else:
        return adjbox(x[x!=0])
    
def human_format(num, round_to= 0):
    '''
    Auxiliary function for number formatting
    '''
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num = round(num / 1000.0, round_to)
    return '{:.{}f}{}'.format(round(num, round_to), round_to, ['', 'K', 'M', 'B', 'T', 'TT'][magnitude])
    
class RCT(object):
    '''
    Evaluates a randomized control trial experiment (A/B test) with 2 groups. By default, these tests are used to compute the mean difference and confidence interval:
    • T-test
    • Bootstrapped means
    • Mann-Whitney U test
    • Bayesian test
    • Difference in difference test
    
    Parameters
    ----------
    data : pandas dataframe containing all of the data
    
    treat_col : str
        Column name of the binary treatment marker. The value 1 marks the treatment and 0 control. To change the treatment/control values, use the optional argument control_val.
    
    tgt_cols : str or list of str
        Column name(s) of the target variables to evaluate. If a variable is detected as binary, some tests do not apply.
        
    alpha : float
        Level of significance for every test. Default 0.05
    
    outliers : str or list of str
        Column name(s) of the target variables where to automatically detect and exclude outlier data. For binary data, no observation is considered outlier. If the data is non-negative, the boxplot method is used in the log-scale for positive values. If the data has negative values, the adjusted boxplot is used for non-zero values. Default is [] (no outlier detection)
    
    split_tgts : str or list of str
        Columns name(s) of target variables to be splitted into binary and continuous parts for evaluation of each. Default [] (no splits)
    
    strat_cols : str or list of str
        Columns name(s) of stratification variables. A separate evaluation is executed for each segment detected, and also for the stratified totals. Default [] (no stratification)
        
    exclude_tests : str or list of str
        Name(s) of the tests to exclude. Can be any of: 't','boo','u','bay','did'. Default [] (no exclusions)
    
    n_boot : int
        Number of bootstrap samples to use in the Bootstrap method. Default 1000
    
    time_col : str
        Column name of the time period counter. The column must be numeric, and the highest value corresponds to the treatment period. If no name is supplied, difference in difference is not calculated. Default None (no DiD)
    
    agg_weight : str
        How to weight the  different strata for aggregation (only used if strat_cols provided). Possible values: 'treat' weights each stratum by the proportion of treated units in that sratum over the total treated units. 'pop' weights by the proportion of total units in that sratum over the total units. Default 'treat'
    
    control_val : same type of 'treat_col' column
        Value that identifies the control group in the treatment column. All other values are assumed as treatment. Default is None, which means that the treatment column is binary and the control value is zero.
    '''
    
    def __init__(self, 
                 data, 
                 treat_col, 
                 tgt_cols, 
                 alpha= 0.05,
                 outliers = [],
                 split_tgts = [],
                 strat_cols= [],
                 exclude_tests= [],
                 n_boot= 1000,
                 time_col= None,
                 agg_weight= 'treat',
                 control_val= None
                ):
        self.data = data
        self.treat_col = treat_col
        self.tgt_cols = [tgt_cols] if type(tgt_cols) == str else tgt_cols.copy()
        assert self.tgt_cols != ['tgt'], 'tgt is a reserved name'
        self.alpha = alpha
        self.outliers = [outliers] if type(outliers) == str else outliers.copy()
        self.split_tgts = [split_tgts] if type(split_tgts) == str else split_tgts.copy()
        self.strat_cols = [strat_cols] if type(strat_cols) == str else strat_cols.copy()
        self.is_strat = len(self.strat_cols) > 0
        self.exclude_tests = [exclude_tests] if type(exclude_tests) == str else exclude_tests.copy()
        self.n_boot = n_boot
        self.time_col = time_col
        self.agg_treat = agg_weight == 'treat'
        self.control_value = control_val
         
        #Seeds for random generators (unique for the data, invariant under row or column permutations)
        self.data_seeds = SeedSequence(np.uint64(pd.util.hash_pandas_object(self.data[sorted(self.data.columns)]).sum())).spawn(2)
    
    def eval_treat(self,
                   verbose= 0
                  ):
        '''
        Runs the specified tests on the data and populates the results dictionary with 2 tables:
            stats: contains general statistics for each sample (size, mean, standard deviation)
            incs: incrementals for each target variable, with confidence intervals

        Parameters
        ----------
        verbose : int
            Control how much information to print:
            1 : print execution time per target and test
            0 : no printing (Default)
        '''
        warnings.simplefilter(action= 'ignore', category= pd.errors.PerformanceWarning)
        
        ''' TODO:
        Loop tgt:
            Eliminar time_col si no queda con data disponible luego de outliers
        '''
        t0 = perf_counter()
        self.times = {'total' : 0.0, 't' : 0.0, 'boo' : 0.0, 'u' : 0.0, 'did' : 0.0, 'bay' : 0.0}
        
        #Add splitted targets
        for c in self.split_tgts:
            self.tgt_cols += [c+'_bin',c+'_cont']
            self.data[c+'_bin'] = self.data[c].astype(bool).astype(np.float32)
            self.data[c+'_cont'] = self.data[c].replace(0.0,np.nan)
            if c in self.outliers:
                self.outliers += [c+'_cont']
        
        #Classify targets
        dist_type = {}
        for tgt in self.tgt_cols:
            if np.array_equal(self.data[tgt].dropna(), self.data[tgt].dropna().astype(bool)):
                dist_type[tgt] = 'binary'
            elif ((self.data[tgt] >= 0).values | self.data[tgt].isna().values).all():
                dist_type[tgt] = 'non-neg'
            else:
                dist_type[tgt] = 'real'
        
        #Build boolean mask to remove nans and outlier and insufficient data
        valid_sel = ~self.data[self.tgt_cols].isna()
        
        #Build treatment column is custom control value is used
        if self.control_value is not None:
            self.data[self.treat_col+'_bin'] = (self.data[self.treat_col] != self.control_value)*1
            self.treat_col = self.treat_col+'_bin'
        
        #Initialize result dataframes
        self.results = {}
        self.results['stats'] = pd.DataFrame(columns= ['tgt','stat',self.treat_col,*self.strat_cols,'value'], 
                                             dtype= np.float64).set_index(['tgt','stat',self.treat_col,*self.strat_cols])
        self.results['incs'] = pd.DataFrame(columns= ['tgt','stat','method',*self.strat_cols,'value'], 
                                             dtype= np.float64).set_index(['tgt','stat','method',*self.strat_cols])
        
        #Check if there is enough data for the evaluation (2 distinct values per segment, treatment marker and period)
        has_time = self.time_col is not None
        time_col_l = [] if not has_time else [self.time_col]
        group_cols = [self.treat_col] + time_col_l + self.strat_cols
        aux = self.data.groupby(group_cols, dropna= False)[self.tgt_cols].apply(lambda x : x.nunique() >= 2)
        if group_cols == [self.treat_col]:
            aux = (aux.min()) & (self.data[self.treat_col].nunique() == 2)
            for t in valid_sel.columns:
                if not aux[t]:
                    valid_sel[t] = False
        else:
            a1 = aux.groupby(time_col_l + self.strat_cols).min()
            aux =  (a1) & ((self.data.groupby(time_col_l + self.strat_cols, dropna= False)[self.treat_col].nunique() == 2).to_numpy().reshape(-1,1))
            valid_sel = self.data[time_col_l + self.strat_cols].join(aux, on= time_col_l + self.strat_cols)[self.tgt_cols] & valid_sel
        
        #Save the post-period value ID
        if has_time:
            post_val = self.data[self.time_col].max()
        
        #Loop through target columns
        for tgt in self.tgt_cols:
            #Pre-select valid rows and target column
            sel_cols = [self.treat_col] + self.strat_cols + time_col_l + [tgt]
            tgt_sel_all = self.data.loc[valid_sel[tgt].values, sel_cols]
            if tgt_sel_all.empty:
                continue
            
            #Detect outliers
            group_cols = [self.treat_col] + self.strat_cols + time_col_l
            if tgt in self.outliers:
                aux = tgt_sel_all.groupby(group_cols, dropna= False)[tgt] \
                    .apply(lambda x : outlier_lims(x, dist_type[tgt])).rename(tgt).to_frame()
                aux[['lo_cut','up_cut']] = pd.DataFrame(aux[tgt].tolist(), index= aux.index)
                out_mask = tgt_sel_all[group_cols].join(aux[['lo_cut','up_cut']], on= group_cols)
                out_mask['out_mask'] = ((tgt_sel_all[tgt] >= out_mask['lo_cut']) & \
                                        (tgt_sel_all[tgt] <= out_mask['up_cut'])) | \
                                        (tgt_sel_all[tgt] == 0)
                out_mask = out_mask['out_mask'].values
            else:
                out_mask = np.ones(len(tgt_sel_all), dtype= bool)
            
            #Last data cleaning and sanity check after outlier elimination
            post_sel = (tgt_sel_all[self.time_col] == post_val).values if has_time else np.ones(len(tgt_sel_all), dtype= bool)
            tgt_sel_post = tgt_sel_all.loc[out_mask & post_sel, :].copy()
            if tgt_sel_post.empty:
                continue
            tgt_sel_all = tgt_sel_all.loc[out_mask, :]
            if has_time:
                tgt_no_hist = not (tgt_sel_all[self.time_col] < post_val).any()
                tgt_sel_post.drop(columns= self.time_col, inplace= True)
            
            aux = tgt_sel_post.groupby([self.treat_col] + self.strat_cols, dropna= False)[tgt].apply(lambda x : x.nunique() >= 2)
            if not self.is_strat:
                if not aux.min():
                    continue
            else:
                aux = aux.groupby(self.strat_cols).min().rename(tgt)
                last_sel = tgt_sel_post[self.strat_cols].join(aux, on= self.strat_cols)[tgt].values
                tgt_sel_post = tgt_sel_post[last_sel]
                if tgt_sel_post.empty:
                    continue
            #Register sample sizes, means and stds
            aux = tgt_sel_post.groupby([self.treat_col] + self.strat_cols, 
                                       dropna= False).aggregate(['count','mean','std']).stack()
            aux.index.set_names('stat', level= -1, inplace= True)
            if self.is_strat:
                #Aggregated sample sizes
                strat_names = aux[tgt].unstack([*self.strat_cols]).columns
                total_tuple = tuple(['total']*strat_names.nlevels) if strat_names.nlevels > 1 else 'total'
                total_tuple_s = tuple(['total']*strat_names.nlevels) if strat_names.nlevels > 1 else ['total']
                aux2_pre = aux[tgt].unstack('stat')
                aux2 = aux2_pre.groupby(self.treat_col)['count'].sum().rename(tgt).to_frame()
                aux2[self.strat_cols] = total_tuple
                aux2['stat'] = 'count'
                aux2.set_index(self.strat_cols + ['stat'], append= True, inplace= True)
                aux = aux.append(aux2)
                
                N1 = aux[tgt].unstack('stat')['count'].loc[1]
                N0 = aux[tgt].unstack('stat')['count'].loc[0]
                Nh = N1.copy() if self.agg_treat else (N1 + N0)
                wh = Nh.drop(total_tuple) / Nh.loc[total_tuple].sum()
                
                #Aggregated means
                aux2 = (aux2_pre['mean'] * wh).groupby(self.treat_col).sum().rename(tgt).to_frame()
                aux2[self.strat_cols] = total_tuple
                aux2['stat'] = 'mean'
                aux2.set_index(self.strat_cols + ['stat'], append= True, inplace= True)
                aux = aux.append(aux2)
                
                #Aggregated stds
                aux2 = (((aux2_pre['std']**2)/(aux2_pre['count']) * (wh**2)) \
                        .groupby(self.treat_col).sum() * (aux2_pre.groupby(self.treat_col)['count'].sum())) \
                        .map(np.sqrt).rename(tgt).to_frame()
                aux2[self.strat_cols] = total_tuple
                aux2['stat'] = 'std'
                aux2.set_index(self.strat_cols + ['stat'], append= True, inplace= True)
                aux = aux.append(aux2)
            else:
                N1 = aux[tgt].unstack('stat')['count'].loc[1]
                N0 = aux[tgt].unstack('stat')['count'].loc[0]
                
            aux['tgt'] = tgt
            aux.set_index(['tgt'], append= True, inplace= True)
            aux.rename(columns= {tgt : 'value'}, inplace= True)
            self.results['stats'] = self.results['stats'].append( \
                aux.reorder_levels(['tgt','stat',self.treat_col,*self.strat_cols]))
            
            #Run the statistical tests (t-test, bootstrap, mannwhitney, diff-in-diff and bayesian)
            if not ('t' in self.exclude_tests):
                t_method = perf_counter()
                #Compute t-test and confidence interval
                dif_mean = (self.results['stats'].loc[(tgt,'mean',1)] - self.results['stats'].loc[(tgt,'mean',0)])['value']
                if self.is_strat:
                    aux_inc = (dif_mean * N1).rename('value').to_frame()
                    aux_inc['stat'] = 'inc'
                    aux_base = (self.results['stats'].loc[(tgt,'mean',0)]['value'] * N1).rename('value').to_frame()
                    aux_base['stat'] = 'base'
                    ttest = tgt_sel_post.groupby(self.strat_cols, dropna= False) \
                    .apply(lambda d : ttest_ind(
                        d.loc[d[self.treat_col] == 1, tgt]
                        , d.loc[d[self.treat_col] == 0, tgt]
                        , equal_var= False))
                    ttest = pd.DataFrame(ttest.tolist(), index= ttest.index).T
                    agg_test = ttest_ind_from_stats(self.results['stats'].loc[(tgt,'mean',1,*total_tuple_s)].squeeze(), 
                                                    self.results['stats'].loc[(tgt,'std',1,*total_tuple_s)].squeeze(), 
                                                    self.results['stats'].loc[(tgt,'count',1,*total_tuple_s)].squeeze(), 
                                                    self.results['stats'].loc[(tgt,'mean',0,*total_tuple_s)].squeeze(), 
                                                    self.results['stats'].loc[(tgt,'std',0,*total_tuple_s)].squeeze(), 
                                                    self.results['stats'].loc[(tgt,'count',0,*total_tuple_s)].squeeze(), 
                                                    equal_var= False)
                    ttest[total_tuple] = list(agg_test)
                    aux_pv = ttest.loc['pvalue'].rename('value').to_frame()
                    aux_pv['stat'] = 'pv'
                    
                    se = dif_mean / ttest.loc['statistic']
                    dof = se**4 / ((self.results['stats'].loc[(tgt,'std',1)]['value']**2 / N1)**2 / (N1 - 1) + 
                                   (self.results['stats'].loc[(tgt,'std',0)]['value']**2 / N0)**2 / (N0 - 1))
                    half_ci = se * N1 * tdist.ppf(1-self.alpha/2, dof)
                    aux_ci_lo = (aux_inc['value'] - half_ci).rename('value').to_frame()
                    aux_ci_lo['stat'] = 'inc_ci_lo'
                    aux_ci_up = (aux_inc['value'] + half_ci).rename('value').to_frame()
                    aux_ci_up['stat'] = 'inc_ci_up'
                    
                    aux = pd.concat([aux_ci_lo, aux_inc, aux_ci_up, aux_pv, aux_base], axis= 0)
                    aux['tgt'] = tgt
                    aux['method'] = 't'
                    aux.set_index(['tgt','stat','method'], append= True, inplace= True)
                    aux = aux.reorder_levels(['tgt','stat','method',*self.strat_cols])
                    self.results['incs'] = self.results['incs'].append(aux)
                else:
                    ttest = ttest_ind(tgt_sel_post.loc[(tgt_sel_post[self.treat_col] == 1), tgt],
                                      tgt_sel_post.loc[(tgt_sel_post[self.treat_col] == 0), tgt], equal_var= False)
                    self.results['incs'].loc[(tgt,'pv','t')] = ttest[1]
                    self.results['incs'].loc[(tgt,'inc','t')] = dif_mean * N1
                    self.results['incs'].loc[(tgt,'base','t')] = self.results['stats'].loc[(tgt,'mean',0)]['value'] * N1
                    se = dif_mean / ttest[0]
                    dof = se**4 / ((self.results['stats'].loc[(tgt,'std',1)]**2 / N1)**2 / (N1 - 1) + 
                                   (self.results['stats'].loc[(tgt,'std',0)]**2 / N0)**2 / (N0 - 1))
                    half_ci = se * N1 * tdist.ppf(1-self.alpha/2, dof)
                    self.results['incs'].loc[(tgt,'inc_ci_up','t')] = self.results['incs'].loc[(tgt,'inc','t')] + half_ci
                    self.results['incs'].loc[(tgt,'inc_ci_lo','t')] = self.results['incs'].loc[(tgt,'inc','t')] - half_ci
                self.times['t'] += perf_counter() - t_method
            if not ('boo' in self.exclude_tests):
                #Compute bootstrap confidence interval
                t_method = perf_counter()
                if dist_type[tgt] != 'binary':
                    dif_mean = (self.results['stats'].loc[(tgt,'mean',1)] - self.results['stats'].loc[(tgt,'mean',0)])['value']
                    if self.is_strat:
                        means_agg = np.zeros((2, self.n_boot, len(strat_names)), dtype= np.float64)
                        strat_seeds = self.data_seeds[0].spawn(len(strat_names))
                        for k, s in enumerate(strat_names):
                            s_s = s if type(s) == tuple else [s]
                            strat_sel = select_strat(tgt_sel_post[self.strat_cols], s)
                            data_cut = tgt_sel_post.loc[strat_sel, [self.treat_col, tgt]]
                            bool_res = boot_eval(data_cut, self.treat_col, tgt, alpha= self.alpha, iters= self.n_boot, 
                                                 seed= strat_seeds[k], return_samples= True)

                            means_agg[1,:,k] = bool_res[4]
                            means_agg[0,:,k] = bool_res[5]

                            self.results['incs'].loc[(tgt,'inc_ci_lo','boo', *s_s)] = bool_res[0] * N1[s]
                            self.results['incs'].loc[(tgt,'inc_ci_up','boo', *s_s)] = bool_res[2] * N1[s]
                            self.results['incs'].loc[(tgt,'pv','boo', *s_s)] = bool_res[3]
                            self.results['incs'].loc[(tgt,'inc','boo', *s_s)] = bool_res[1] * N1[s]
                            self.results['incs'].loc[(tgt,'base','boo', *s_s)] = \
                                self.results['stats'].loc[(tgt,'mean',0,*s_s),'value'] * N1[s]

                        means_tot = (means_agg * wh.to_numpy().reshape((1,1,-1))).sum(axis= 2)
                        del means_agg
                        diffs = means_tot[1,:] - means_tot[0,:]
                        ci_lims = np.quantile(diffs, [self.alpha/2, 1-self.alpha/2])
                        inc = dif_mean.loc[total_tuple].squeeze()
                        pv = ((diffs <= (inc - abs(inc))) | (diffs >= (inc + abs(inc)))).mean()
                        self.results['incs'].loc[(tgt,'inc_ci_lo','boo', *total_tuple_s)] = \
                            ci_lims[0] * N1[total_tuple].squeeze()
                        self.results['incs'].loc[(tgt,'inc_ci_up','boo', *total_tuple_s)] = \
                            ci_lims[1] * N1[total_tuple].squeeze()
                        self.results['incs'].loc[(tgt,'pv','boo', *total_tuple_s)] = pv
                        self.results['incs'].loc[(tgt,'inc','boo', *total_tuple_s)] = \
                            (dif_mean.loc[total_tuple] * N1[total_tuple]).squeeze()
                        self.results['incs'].loc[(tgt,'base','boo', *total_tuple_s)] = \
                            (self.results['stats'].loc[(tgt,'mean',0,*total_tuple_s),'value'] * N1[total_tuple]).squeeze()
                        del means_tot, diffs
                    else:
                        data_cut = tgt_sel_post[[self.treat_col, tgt]]
                        bool_res = boot_eval(data_cut, self.treat_col, tgt, alpha= self.alpha, iters= self.n_boot, 
                                             seed= self.data_seeds[0])
                        self.results['incs'].loc[(tgt,'inc_ci_lo','boo')] = bool_res[0] * N1
                        self.results['incs'].loc[(tgt,'inc_ci_up','boo')] = bool_res[2] * N1
                        self.results['incs'].loc[(tgt,'pv','boo')] = bool_res[3]
                        self.results['incs'].loc[(tgt,'inc','boo')] = bool_res[1] * N1
                        self.results['incs'].loc[(tgt,'base','boo')] = self.results['stats'].loc[(tgt,'mean',0),'value'] * N1
                else:
                    #For binary data, the Fisher test is used instead of the Bootstrap
                    if self.is_strat:
                        for k, s in enumerate(strat_names):
                            s_s = s if type(s) == tuple else [s]
                            strat_sel = select_strat(tgt_sel_post[self.strat_cols], s)
                            data_cut = tgt_sel_post.loc[strat_sel, [self.treat_col, tgt]]
                            fish_res = fisher_eval(data_cut, self.treat_col, tgt, alpha= self.alpha)

                            self.results['incs'].loc[(tgt,'inc_ci_lo','boo', *s_s)] = fish_res[0] * N1[s]
                            self.results['incs'].loc[(tgt,'inc_ci_up','boo', *s_s)] = fish_res[2] * N1[s]
                            self.results['incs'].loc[(tgt,'pv','boo', *s_s)] = fish_res[3]
                            self.results['incs'].loc[(tgt,'inc','boo', *s_s)] = fish_res[1] * N1[s]
                            self.results['incs'].loc[(tgt,'base','boo', *s_s)] = \
                                self.results['stats'].loc[(tgt,'mean',0,*s_s),'value'] * N1[s]
                        data_cut = tgt_sel_post[[self.treat_col, tgt]]
                        fish_res = fisher_eval(data_cut, self.treat_col, tgt, alpha= self.alpha)
                        self.results['incs'].loc[(tgt,'inc_ci_lo','boo', *total_tuple_s)] = \
                            fish_res[0] * N1[total_tuple].squeeze()
                        self.results['incs'].loc[(tgt,'inc_ci_up','boo', *total_tuple_s)] = \
                            fish_res[2] * N1[total_tuple].squeeze()
                        self.results['incs'].loc[(tgt,'pv','boo', *total_tuple_s)] = fish_res[3]
                        self.results['incs'].loc[(tgt,'inc','boo', *total_tuple_s)] = \
                            (fish_res[1] * N1[total_tuple]).squeeze()
                        self.results['incs'].loc[(tgt,'base','boo', *total_tuple_s)] = \
                            (self.results['stats'].loc[(tgt,'mean',0,*total_tuple_s),'value'] * N1[total_tuple]).squeeze()
                    else:
                        data_cut = tgt_sel_post[[self.treat_col, tgt]]
                        fish_res = fisher_eval(data_cut, self.treat_col, tgt, alpha= self.alpha)
                        self.results['incs'].loc[(tgt,'inc_ci_lo','boo')] = fish_res[0] * N1
                        self.results['incs'].loc[(tgt,'inc_ci_up','boo')] = fish_res[2] * N1
                        self.results['incs'].loc[(tgt,'pv','boo')] = fish_res[3]
                        self.results['incs'].loc[(tgt,'inc','boo')] = fish_res[1] * N1
                        self.results['incs'].loc[(tgt,'base','boo')] = self.results['stats'].loc[(tgt,'mean',0),'value'] * N1
                self.times['boo'] += perf_counter() - t_method
                
            if not ('u' in self.exclude_tests or dist_type[tgt] == 'binary'):
                #Compute mann-whitney u-test and confidence interval
                t_method = perf_counter()
                if self.is_strat:
                    for s in strat_names:
                        s_s = s if type(s) == tuple else [s]
                        strat_sel = select_strat(tgt_sel_post[self.strat_cols], s)
                        data_cut = tgt_sel_post.loc[strat_sel, [self.treat_col, tgt]]
                        if (data_cut[self.treat_col] == 0).sum() < 20:
                            mannu_res = tuple([np.nan]*6)
                        else:
                            mannu_res = mannu_prop_eval(data_cut, self.treat_col, tgt, alpha= self.alpha)
                        self.results['incs'].loc[(tgt,'inc','u', *s_s)] = mannu_res[1] * N1[s]
                        self.results['incs'].loc[(tgt,'base','u', *s_s)] = \
                            (self.results['stats'].loc[(tgt,'mean',1,*s_s)] - mannu_res[1]) * N1[s]
                        self.results['incs'].loc[(tgt,'pv','u', *s_s)] = mannu_res[3]
                        self.results['incs'].loc[(tgt,'inc_ci_up','u', *s_s)] = mannu_res[2] * N1[s]
                        self.results['incs'].loc[(tgt,'inc_ci_lo','u', *s_s)] = mannu_res[0] * N1[s]
                        #self.results['incs'].loc[(tgt,'auc_up','u'), s] = mannu_res[5]
                        #self.results['incs'].loc[(tgt,'auc_lo','u'), s] = mannu_res[4]
                    if not self.results['incs'].loc[(tgt,'inc','u')]['value'].isna().all():
                        self.results['incs'].loc[(tgt,'inc','u', *total_tuple_s)] = \
                            (self.results['incs'].loc[(tgt,'inc','u')]['value'] / N1[strat_names] * wh).sum() \
                            * N1[total_tuple].squeeze()
                        self.results['incs'].loc[(tgt,'inc_ci_up','u', *total_tuple_s)] = \
                            np.sqrt((((self.results['incs'].loc[(tgt,'inc_ci_up','u')]['value'] - \
                            self.results['incs'].loc[(tgt,'inc','u')].drop(total_tuple)['value']) / N1[strat_names] * wh)**2).sum()) \
                            * N1[total_tuple].squeeze() + self.results['incs'].loc[(tgt,'inc','u',*total_tuple_s)]['value'].squeeze()
                        self.results['incs'].loc[(tgt,'inc_ci_lo','u', *total_tuple_s)] = \
                            self.results['incs'].loc[(tgt,'inc','u', *total_tuple_s)]['value'].squeeze() - \
                            np.sqrt((((self.results['incs'].loc[(tgt,'inc','u')].drop(total_tuple)['value'] - \
                            self.results['incs'].loc[(tgt,'inc_ci_lo','u')]['value']) / \
                                      N1[strat_names] * wh)**2).sum()) * N1[total_tuple].squeeze()
                        self.results['incs'].loc[(tgt,'base','u', *total_tuple_s)] = \
                            self.results['stats'].loc[(tgt,'mean',1,*total_tuple_s)]['value'].squeeze() * N1[total_tuple].squeeze() - \
                            self.results['incs'].loc[(tgt,'inc','u', *total_tuple_s)]['value'].squeeze()
                        Z = norm.ppf(1 - self.alpha/2)
                        sigma_up = (self.results['incs'].loc[(tgt,'inc_ci_up','u', *total_tuple_s)]['value'].squeeze() - \
                                    self.results['incs'].loc[(tgt,'inc','u', *total_tuple_s)]['value'].squeeze()) / Z
                        sigma_lo = (self.results['incs'].loc[(tgt,'inc','u', *total_tuple_s)]['value'].squeeze() - \
                                    self.results['incs'].loc[(tgt,'inc_ci_lo','u', *total_tuple_s)]['value'].squeeze()) / Z
                        inc = self.results['incs'].loc[(tgt,'inc','u', *total_tuple_s)]['value'].squeeze()
                        pv = norm.cdf(inc - abs(inc), scale= sigma_lo) + norm.cdf(-inc - abs(inc), scale= sigma_up)
                        self.results['incs'].loc[(tgt,'pv','u', *total_tuple_s)] = pv
                    else:
                        self.results['incs'].loc[(tgt,'inc','u', *total_tuple_s)] = np.nan
                        self.results['incs'].loc[(tgt,'inc_ci_up','u', *total_tuple_s)] = np.nan
                        self.results['incs'].loc[(tgt,'inc_ci_lo','u', *total_tuple_s)] = np.nan
                        self.results['incs'].loc[(tgt,'base','u', *total_tuple_s)] = np.nan
                        self.results['incs'].loc[(tgt,'pv','u', *total_tuple_s)] = np.nan
                else:
                    data_cut = tgt_sel_post[[self.treat_col, tgt]]
                    if (data_cut[self.treat_col] == 0).sum() < 20:
                        mannu_res = tuple([np.nan]*6)
                    else:
                        mannu_res = mannu_prop_eval(data_cut, self.treat_col, tgt, alpha= self.alpha)
                    self.results['incs'].loc[(tgt,'inc','u')] = mannu_res[1] * N1
                    self.results['incs'].loc[(tgt,'base','u')] = \
                        (self.results['stats'].loc[(tgt,'mean',1),'value'] - mannu_res[1]) * N1
                    self.results['incs'].loc[(tgt,'pv','u')] = mannu_res[3]
                    self.results['incs'].loc[(tgt,'inc_ci_up','u')] = mannu_res[2] * N1
                    self.results['incs'].loc[(tgt,'inc_ci_lo','u')] = mannu_res[0] * N1
                    #self.results['incs'].loc[(tgt,'auc_up','u')] = mannu_res[5]
                    #self.results['incs'].loc[(tgt,'auc_lo','u')] = mannu_res[4]
                self.times['u'] += perf_counter() - t_method
            
            if not ('did' in self.exclude_tests or self.time_col is None or tgt_no_hist):
                #Compute difference-in-difference confidence interval
                t_method = perf_counter()
                if self.is_strat:
                    for s in strat_names:
                        s_s = s if type(s) == tuple else [s]
                        strat_sel = select_strat(tgt_sel_all[self.strat_cols], s)
                        features = pd.DataFrame(columns= ['constant', 't', 'post', 'gt', 'pgt'], dtype= np.float64)
                        features['t'] = tgt_sel_all.loc[strat_sel, self.time_col].astype(float)
                        features['post'] = ((tgt_sel_all[self.time_col] == post_val).values)[strat_sel].astype(float)
                        features['gt'] = tgt_sel_all.loc[strat_sel, self.treat_col].astype(float)
                        features['pgt'] = features['post'] * features['gt']
                        features['constant'] = 1.0
                        if not (tgt_sel_all.loc[strat_sel, self.time_col].nunique() > 2):
                            features.drop(columns= 't', inplace= True)
                        has_post = True
                        if not (features['post'] != 1.0).any():
                            features.drop(columns= ['post','gt'], inplace= True)
                            has_post = False
                        model = OLS(tgt_sel_all.loc[strat_sel, tgt].astype(float), features, hasconst= True)
                        did_res = model.fit()

                        inc = did_res.params['pgt']
                        inc_ci = did_res.conf_int(alpha= self.alpha, cols= None).loc['pgt']
                        if has_post:
                            bias = did_res.params['gt']
                        else:
                            bias = np.nan
                        pv = did_res.pvalues['pgt']

                        self.results['incs'].loc[(tgt,'inc','did', *s_s)] = inc * N1[s]
                        self.results['incs'].loc[(tgt,'inc_ci_up','did', *s_s)] = inc_ci[1] * N1[s]
                        self.results['incs'].loc[(tgt,'inc_ci_lo','did', *s_s)] = inc_ci[0] * N1[s]
                        self.results['incs'].loc[(tgt,'pv','did', *s_s)] = pv
                        self.results['incs'].loc[(tgt,'pre_bias','did', *s_s)] = bias * N1[s]
                        self.results['incs'].loc[(tgt,'base','did', *s_s)] = \
                            (self.results['stats'].loc[(tgt,'mean',1,*s_s),'value'] - inc) * N1[s]

                    features = pd.DataFrame(columns= ['constant', 't', 'post', 'gt', 'pgt'], dtype= np.float64)
                    features['t'] = tgt_sel_all[self.time_col].astype(float)
                    features['post'] = (tgt_sel_all[self.time_col] == post_val).values.astype(float)
                    features['gt'] = tgt_sel_all[self.treat_col].astype(float)
                    features['pgt'] = features['post'] * features['gt']
                    features['constant'] = 1.0
                    if not (tgt_sel_all[self.time_col].nunique() > 2):
                        features.drop(columns= 't', inplace= True)
                    has_post = True
                    if not (features['post'] != 1.0).any():
                        features.drop(columns= ['post','gt'], inplace= True)
                        has_post = False
                    for k, s in enumerate(strat_names):
                        s_s = s if type(s) == tuple else [s]
                        strat_sel = select_strat(tgt_sel_all[self.strat_cols], s)
                        strat_feat_name = 'strat_onehot_'+str(k)
                        features[strat_feat_name] = 0.0
                        features.loc[strat_sel, strat_feat_name] = 1.0
                        if k == len(strat_names) - 1:
                            features.drop(columns= strat_feat_name, inplace= True)
                    join_cols = [self.treat_col,self.time_col]+self.strat_cols
                    n_strats = tgt_sel_all.groupby(join_cols, dropna= False)[tgt].count()
                    if self.agg_treat:
                        wh_all = n_strats.loc[1]
                    else:
                        wh_all = n_strats.groupby([self.time_col]+self.strat_cols, dropna= False).sum()
                    wh_all = wh_all / wh_all.groupby(self.time_col).sum()
                    n_treats = n_strats.groupby([self.treat_col,self.time_col], dropna= False).sum()
                    weights = wh_all / n_strats * n_treats
                    weights = tgt_sel_all[join_cols].join(weights.rename('weights').reorder_levels(join_cols), on= join_cols)
                    weights = weights['weights'].to_numpy()
                    model = WLS(tgt_sel_all[tgt].astype(float), features, weights= weights, hasconst= True)
                    did_res = model.fit()

                    inc = did_res.params['pgt']
                    inc_ci = did_res.conf_int(alpha= self.alpha, cols= None).loc['pgt']
                    if has_post:
                        bias = did_res.params['gt']
                    else:
                        bias = np.nan
                    pv = did_res.pvalues['pgt']

                    self.results['incs'].loc[(tgt,'inc','did', *total_tuple_s)] = inc * N1[total_tuple].squeeze()
                    self.results['incs'].loc[(tgt,'inc_ci_up','did', *total_tuple_s)] = inc_ci[1] * N1[total_tuple].squeeze()
                    self.results['incs'].loc[(tgt,'inc_ci_lo','did', *total_tuple_s)] = inc_ci[0] * N1[total_tuple].squeeze()
                    self.results['incs'].loc[(tgt,'pv','did', *total_tuple_s)] = pv
                    self.results['incs'].loc[(tgt,'pre_bias','did', *total_tuple_s)] = bias * N1[total_tuple].squeeze()
                    self.results['incs'].loc[(tgt,'base','did', *total_tuple_s)] = \
                        (self.results['stats'].loc[(tgt,'mean',1, *total_tuple_s),'value'].squeeze() - inc) * N1[total_tuple].squeeze()

                else:
                    features = pd.DataFrame(columns= ['constant', 't', 'post', 'gt', 'pgt'], dtype= np.float64)
                    features['t'] = tgt_sel_all[self.time_col].astype(float)
                    features['post'] = (tgt_sel_all[self.time_col] == post_val).values.astype(float)
                    features['gt'] = tgt_sel_all[self.treat_col].astype(float)
                    features['pgt'] = features['post'] * features['gt']
                    features['constant'] = 1.0
                    if not (tgt_sel_all[self.time_col].nunique() > 2):
                        features.drop(columns= 't', inplace= True)
                    model = OLS(tgt_sel_all[tgt].astype(float), features, hasconst= True)
                    did_res = model.fit()

                    inc = did_res.params['pgt']
                    inc_ci = did_res.conf_int(alpha= self.alpha, cols= None).loc['pgt']
                    bias = did_res.params['gt']
                    pv = did_res.pvalues['pgt']

                    self.results['incs'].loc[(tgt,'inc','did')] = inc * N1
                    self.results['incs'].loc[(tgt,'base','did')] = \
                        (self.results['stats'].loc[(tgt,'mean',1),'value'] - inc) * N1
                    self.results['incs'].loc[(tgt,'inc_ci_up','did')] = inc_ci[1] * N1
                    self.results['incs'].loc[(tgt,'inc_ci_lo','did')] = inc_ci[0] * N1
                    self.results['incs'].loc[(tgt,'pv','did')] = pv
                    self.results['incs'].loc[(tgt,'pre_bias','did')] = bias * N1
                self.times['did'] += perf_counter() - t_method
                
            if not ('bay' in self.exclude_tests):
                #Compute bayesian confidence interval
                t_method = perf_counter()
                if self.is_strat:
                    means_agg = np.empty((len(strat_names), 100000, 2), dtype= np.float64)
                    strat_seeds = self.data_seeds[1].spawn(len(strat_names))
                    for k, s in enumerate(strat_names):
                        s_s = s if type(s) == tuple else [s]
                        strat_sel = select_strat(tgt_sel_post[self.strat_cols], s)
                        data_cut = tgt_sel_post.loc[strat_sel, [self.treat_col, tgt]]
                        bay_res = bayes_smoothing_eval(data_cut, self.treat_col, tgt, 
                                                       alpha= self.alpha, seed= strat_seeds[k], return_samples= True)
                        self.results['incs'].loc[(tgt,'inc','bay', *s_s)] = bay_res[1] * N1[s]
                        self.results['incs'].loc[(tgt,'inc_ci_up','bay', *s_s)] = bay_res[2] * N1[s]
                        self.results['incs'].loc[(tgt,'inc_ci_lo','bay', *s_s)] = bay_res[0] * N1[s]
                        self.results['incs'].loc[(tgt,'base','bay', *s_s)] = \
                            (self.results['stats'].loc[(tgt,'mean',1,*s_s),'value'] - bay_res[1]) * N1[s]
                        self.results['incs'].loc[(tgt,'pv','bay', *s_s)] = bay_res[3]
                        means_agg[k,:,1] = bay_res[6]
                        means_agg[k,:,0] = bay_res[7]

                    means_tot = (means_agg * wh.to_numpy().reshape((-1,1,1))).sum(axis= 0)
                    del means_agg
                    diffs = means_tot[:,1] - means_tot[:,0]
                    inc = diffs.mean()
                    ci_lims = np.quantile(diffs, [self.alpha/2, 1-self.alpha/2])
                    pv = ((diffs <= (inc - abs(inc))) | (diffs >= (inc + abs(inc)))).mean()
                    self.results['incs'].loc[(tgt,'inc','bay', *total_tuple_s)] = inc * N1[total_tuple].squeeze()
                    self.results['incs'].loc[(tgt,'inc_ci_lo','bay', *total_tuple_s)] = ci_lims[0] * N1[total_tuple].squeeze()
                    self.results['incs'].loc[(tgt,'inc_ci_up','bay', *total_tuple_s)] = ci_lims[1] * N1[total_tuple].squeeze()
                    self.results['incs'].loc[(tgt,'base','bay', *total_tuple_s)] = \
                        self.results['stats'].loc[(tgt,'mean',1,*total_tuple_s),'value'].squeeze() * N1[total_tuple].squeeze() - \
                        self.results['incs'].loc[(tgt,'inc','bay',*total_tuple_s),'value'].squeeze()
                    self.results['incs'].loc[(tgt,'pv','bay', *total_tuple_s)] = pv
                    del means_tot, diffs
                else:
                    bay_res = bayes_smoothing_eval(tgt_sel_post, self.treat_col, tgt, 
                                                   alpha= self.alpha, seed= self.data_seeds[1])
                    self.results['incs'].loc[(tgt,'inc','bay')] = bay_res[1] * N1
                    self.results['incs'].loc[(tgt,'base','bay')] = \
                        (self.results['stats'].loc[(tgt,'mean',1),'value'] - bay_res[1]) * N1
                    self.results['incs'].loc[(tgt,'inc_ci_up','bay')] = bay_res[2] * N1
                    self.results['incs'].loc[(tgt,'inc_ci_lo','bay')] = bay_res[0] * N1
                    self.results['incs'].loc[(tgt,'pv','bay')] = bay_res[3]
                self.times['bay'] += perf_counter() - t_method
            #Drop added splitted columns
            if (tgt.endswith('_bin') and tgt[:-4] in self.split_tgts) or \
                (tgt.endswith('_cont') and tgt[:-5] in self.split_tgts):
                self.data.drop(columns= tgt, inplace= True)
            
        self.results['stats'].sort_index(inplace= True)
        self.results['incs'].sort_index(inplace= True)
        if self.is_strat:
            self.results['stats'] = self.results['stats'].value.unstack(self.strat_cols)
            self.results['incs'] = self.results['incs'].value.unstack(self.strat_cols)
        else:
            self.results['stats'] = self.results['stats'].rename(columns= {'value' : 'total'})
            self.results['incs'] = self.results['incs'].rename(columns= {'value' : 'total'})
        if self.control_value is not None:
            self.data.drop(columns= self.treat_col, inplace= True)
            self.treat_col = self.treat_col[:-4]
        self.dist_type = dist_type
        self.times['total'] += perf_counter() - t0
        if verbose:
            print('Execution times:')
            for k,v in self.times.items():
                print(f'{k}: {v : 0.1f} seconds')
    
    def plot(self):
        names_methods = {'t' : 'T test', 'u' : 'U test', 'boo' : 'Bootstrap', 'bay' : 'Bayesian', 'did' : 'Dif in Dif'}
        num_tgts = len(self.tgt_cols)
        for seg in self.results['incs'].columns:
            res_stats = self.results['stats'][seg].squeeze()
            res_incs = self.results['incs'][seg].squeeze()
            
            plt.figure(figsize= [6.667 * num_tgts,5])
            for j, tgt in enumerate(self.tgt_cols):
                #Select methods used
                method_list = ['t','boo'] + (['bay'] if self.dist_type[tgt] == 'binary' else ['u','bay']) + \
                                            ([] if self.time_col is None else ['did'])
                methods = [m for m in method_list if not m in self.exclude_tests]
                if self.dist_type[tgt] == 'binary' and 'u' in methods:
                    methods.remove('u')
                if self.time_col is None and 'did' in methods:
                    methods.remove('did')
                #Extract incrementals and intervals
                cis = np.zeros((2,len(methods)))
                incs = np.zeros(len(methods))
                for k, m in enumerate(methods):
                    incs[k] = res_incs.loc[(tgt,'inc',m)]
                    cis[0, k] = incs[k] - res_incs.loc[(tgt,'inc_ci_lo',m)]
                    cis[1, k] = res_incs.loc[(tgt,'inc_ci_up',m)] - incs[k] 
                range_n = [k for k, m in enumerate(method_list) if m in methods]
                #Precalculations
                total_gt = res_stats.loc[(tgt,'mean',1)] * (res_stats.loc[(tgt,'count',1)] \
                                                             if self.dist_type[tgt] != 'binary' else 1.0)
                total_gc = res_stats.loc[(tgt,'mean',0)] * (res_stats.loc[(tgt,'count',0)] \
                                                             if self.dist_type[tgt] != 'binary' else 1.0)
                total_gt = human_format(total_gt) if self.dist_type[tgt] != 'binary' else f'{total_gt:.1%}'
                total_gc = human_format(total_gc) if self.dist_type[tgt] != 'binary' else f'{total_gc:.1%}'
                #Title
                plt.subplot(1, num_tgts, j+1)
                seg_name = 'Seg ' + str(seg) + ', ' \
                            if seg != (tuple(['total']*len(seg)) \
                                if type(seg) == tuple else 'total') else 'Total '
                plt.title(seg_name + tgt + f' (GT: {total_gt}, GC: {total_gc})', fontsize= 16)
                #Errorbars
                plt.errorbar(range_n, incs, yerr= cis, mew= 4, capsize= 5.0, fmt= '.')
                #Visual formatting
                plt.gca().yaxis.set_visible(False)
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['bottom'].set_visible(False)
                plt.gca().spines['left'].set_visible(False)
                plt.gca().set_xticks(list(range(len(method_list))))
                lab_names = [names_methods[m] for m in method_list]
                if self.dist_type[tgt] == 'binary' and 'Bootstrap' in lab_names:
                    lab_names = [(s if s != 'Bootstrap' else 'Fisher') for s in lab_names]
                plt.gca().set_xticklabels(lab_names, fontsize= 14)
                plt.tick_params(axis= 'x', which= 'both', bottom= False, top= False) 
                #Zero line
                plt.axhline(0.0, color= (1,0,0,0.7))
                #Labels with incremental values and %
                bases = res_incs.loc[tgt].loc['base']
                for k, i in enumerate(incs):
                    plt.gca().annotate(human_format(i), (range_n[k] - 0.07, i), horizontalalignment= 'right', 
                                       verticalalignment= 'center', fontweight='bold', fontsize= 12)
                    plt.gca().annotate(human_format(i-cis[0,k]), (range_n[k] - 0.07, i - cis[0,k]), 
                                       horizontalalignment= 'right', verticalalignment= 'center', fontsize= 12)
                    plt.gca().annotate(human_format(i+cis[1,k]), (range_n[k] - 0.07, i + cis[1,k]), 
                                       horizontalalignment= 'right', verticalalignment= 'center', fontsize= 12)
                    base = bases[methods[k]]
                    plt.gca().annotate(f'{i / base:.0%}', (range_n[k] + 0.07, i), horizontalalignment= 'left', 
                                       verticalalignment= 'center', fontweight='bold', fontsize= 12)
                    plt.gca().annotate(f'{(i-cis[0,k]) / base:.0%}', (range_n[k] + 0.07, i - cis[0,k]), 
                                       horizontalalignment= 'left', verticalalignment= 'center', fontsize= 12)
                    plt.gca().annotate(f'{(i+cis[1,k]) / base:.0%}', (range_n[k] + 0.07, i + cis[1,k]), 
                                       horizontalalignment= 'left', verticalalignment= 'center', fontsize= 12)
                plt.gca().annotate(f'{0.0:.1g}', (- 0.15, 0.0), horizontalalignment= 'left', verticalalignment= 'bottom', 
                                   fontweight='bold', fontsize= 12)
            plt.savefig('./static/output.jpg')
    
def pre_exp_tt_totalpop(cont_prop, inc_prop, avg, std= None, alpha= 0.05, power= 0.8):
    '''
    Computes the required total number of observations in an A/B test to detect an effect with a t-test
    
    Parameters
    ----------
    cont_prop : float
        Proportion of total population that will be in the control group. Must be between (but not including) 0 and 1.
    
    inc_prop : float
        Expected ratio between the mean difference to be detected and the control mean. Must be greater than zero.
        
    avg : float
        Expected sample average in the control group. Cannot be zero.
        
    std: float
        Expected pooled standard deviation. Must be greater than zero. If not provided, it is assumed that the target variable is binary and std = sqrt(avg*(1-avg))
        
    alpha : float
        Level of significance to be used when applying the t-test after the experiment. Default is 0.05
        
    power : float
        Requested probability of detecting the implied effect (1 - Type II error probability). Default is 0.8
        
    Returns
    ----------
    
    total_pop : int
        Minimum total number of observations required to detect the given effect
    '''
    if std is None:
        assert avg > 0 and avg < 1, 'If standard deviation is not provided, the average must be in the interval (0,1)'
        std = np.sqrt(avg*(1-avg))
    assert inc_prop > 0, 'Expected incremental must be a proportion greater than 0'
    assert std > 0, 'Standard deviation must be greater than 0'
    assert avg > 0 or avg < 0, 'Target average cannot be 0'
    efs = abs(avg) * inc_prop / std
    assert cont_prop > 0 and cont_prop < 1, 'Control proportion must be in the interval (0,1)'
    ratio = (1 - cont_prop) / cont_prop
    nobs2 = int(np.ceil(tt_ind_solve_power(effect_size= efs, alpha= alpha, power= power, ratio= ratio)))
    nobs1 = int(np.ceil(nobs2 * ratio))
    return nobs1 + nobs2
    
def pre_exp_tt_contprop(total_pop, inc_prop, avg, std= None, alpha= 0.05, power= 0.8):
    '''
    Computes the required proportion of control observations in an A/B test to detect an effect with a t-test
    
    Parameters
    ----------
    total_pop : int
        Total number of observations available
    
    inc_prop : float
        Expected ratio between the mean difference to be detected and the control mean. Must be greater than zero.
        
    avg : float
        Expected sample average in the control group. Cannot be zero.
        
    std: float
        Expected pooled standard deviation. Must be greater than zero. If not provided, it is assumed that the target variable is binary and std = sqrt(avg*(1-avg))
        
    alpha : float
        Level of significance to be used when applying the t-test after the experiment. Default is 0.05
        
    power : float
        Requested probability of detecting the implied effect (1 - Type II error probability). Default is 0.8
        
    Returns
    ----------
    
    cont_prop : float
        Minimum proportion of control observations required to detect the given effect. If there are not enough observations to detect the effect, the maximum control is returned (50%). In this case, the resulting power could be lower than 80%.
    '''
    root_func = lambda cont_prop : pre_exp_tt_totalpop(cont_prop, inc_prop, avg, std= std, alpha= alpha, power= power) - total_pop
    if root_func(0.5) >= 0:
        return 0.5
    return root_scalar(root_func, method= 'brentq', bracket= (0.5, 1/total_pop), rtol= 0.01, maxiter= 100).root

def pre_exp_tt_pwr(total_pop, cont_prop, inc_prop, avg, std= None, alpha= 0.05):
    '''
    Computes the resulting power (probability of detecting an expected effect) given an experiment configuration.
    
    Parameters
    ----------
    total_pop : int
        Total number of observations available
    
    cont_prop : float
        Proportion of total population that will be in the control group. Must be between (but not including) 0 and 1.
    
    inc_prop : float
        Expected ratio between the mean difference to be detected and the control mean. Must be greater than zero.
        
    avg : float
        Expected sample average in the control group. Cannot be zero.
        
    std: float
        Expected pooled standard deviation. Must be greater than zero. If not provided, it is assumed that the target variable is binary and std = sqrt(avg*(1-avg))
        
    alpha : float
        Level of significance to be used when applying the t-test after the experiment. Default is 0.05
        
    Returns
    ----------
    
    power : float
        Probability of detecting the implied effect (1 - Type II error probability). Default is 0.8
    '''
    if std is None:
        assert avg > 0 and avg < 1, 'If standard deviation is not provided, the average must be in the interval (0,1)'
        std = np.sqrt(avg*(1-avg))
    assert inc_prop > 0, 'Expected incremental must be a proportion greater than 0'
    assert std > 0, 'Standard deviation must be greater than 0'
    assert avg > 0 or avg < 0, 'Target average cannot be 0'
    efs = abs(avg) * inc_prop / std
    assert cont_prop > 0 and cont_prop < 1, 'Control proportion must be in the interval (0,1)'
    ratio = (1 - cont_prop) / cont_prop
    nobs1 = total_pop * cont_prop
    return efs, tt_ind_solve_power(effect_size= efs, alpha= alpha, nobs1= nobs1, ratio= ratio)

def pre_exp_tt_plot_incpwr(total_pop, cont_prop, inc_prop, avg, std= None, alpha= 0.05, ret_values= False):
    '''
    Plots the resulting power for different values of the expected incremental proportion.
    
    Parameters
    ----------
    total_pop : int
        Total number of observations available
    
    cont_prop : float
        Proportion of total population that will be in the control group. Must be between (but not including) 0 and 1.
    
    inc_prop : float
        Expected ratio between the mean difference to be detected and the control mean. Must be greater than zero.
        
    avg : float
        Expected sample average in the control group. Cannot be zero.
        
    std: float
        Expected pooled standard deviation. Must be greater than zero. If not provided, it is assumed that the target variable is binary and std = sqrt(avg*(1-avg))
        
    alpha : float
        Level of significance to be used when applying the t-test after the experiment. Default is 0.05
    
    ret_values : bool
        Returns a dataframe with the plotted values if True. No return if False.
    Returns
    ----------
    
    df : pandas DataFrame
        Plotted values. Only returned if ret_values is True
    '''
    incs = [inc_prop * 0.9 ** n for n in range(-5, 6)]
    betas = [pre_exp_tt_pwr(total_pop, cont_prop, i, avg, std=std, alpha=alpha)[1] for i in incs]
    df = pd.DataFrame({'% Incremental': incs, 'Power': betas})
    df *= 100
    df.set_index('% Incremental', inplace=True)
    df.plot()
    plt.axhline(betas[5] * 100, color=(1, 0, 0, 0.5), linestyle='dashed')
    plt.legend(['Power', 'Base Value'])
    plt.axvline(incs[5] * 100, color=(1, 0, 0, 0.5), linestyle='dashed')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.title('Power vs % Incremental \nSensitivity analysis', fontsize=14)
    plt.ylabel('% Power')
    plt.savefig('./static/power.jpg')
    if ret_values:
        return df

