"""This module contains functions for calculating the acquisition score of an input based on
various metrics"""
from typing import Callable, Optional, Set

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm, rankdata

RG = np.random.default_rng()


def set_seed(seed: Optional[int] = None) -> None:
    """Set the seed of this module's random number generator"""
    global RG
    RG = np.random.default_rng(seed)


def get_metric(metric: str) -> Callable[..., np.ndarray]:
    """Get the corresponding metric function"""
    try:
        return {
            "random": random,
            "threshold": threshold,
            "greedy": greedy,
            "noisy": noisy,
            "ucb": ucb,
            "lcb": lcb,
            "thompson": thompson,
            "ts": thompson,
            "ei": ei,
            "pi": pi,
            "md_ei": md_ei,
        }[metric]
    except KeyError:
        raise ValueError(f'Unrecognized metric: "{metric}"')


def get_needs(metric: str) -> Set:
    """Get the values needed to compute this metric"""
    return {
        "random": set(),
        "greedy": {"means"},
        "noisy": {"means"},
        "ucb": {"means", "vars"},
        "ei": {"means", "vars"},
        "pi": {"means", "vars"},
        "thompson": {"means", "vars"},
        "ts": {"means", "vars"},
        "threshold": {"means"},
        "md_ei": {"means", "vars"},
    }.get(metric, set())


def valid_metrics() -> Set[str]:
    return {"random", "threshold", "greedy", "noisy", "ucb", "lcb", "ts", "thompson", "ei", "pi", "md_ei"}


def calc(
    metric: str,
    Y_mean: np.ndarray,
    Y_var: np.ndarray,
    current_max: float,
    t: float,
    beta: int,
    xi: float,
    stochastic: bool,
) -> np.ndarray:
    """Call corresponding metric function with the proper args"""
    if metric == "random":
        return random(Y_mean)
    if metric == "threshold":
        return threshold(Y_mean, t)
    if metric == "greedy":
        return greedy(Y_mean)
    if metric == "noisy":
        return noisy(Y_mean)
    if metric == "ucb":
        return ucb(Y_mean, Y_var, beta)
    if metric == "lcb":
        return lcb(Y_mean, Y_var, beta)
    if metric in ["ts", "thompson"]:
        return thompson(Y_mean, Y_var, stochastic)
    if metric == "ei":
        return ei(Y_mean, Y_var, current_max, xi)
    if metric == "pi":
        return pi(Y_mean, Y_var, current_max, xi)

    raise ValueError(f'Unrecognized metric "{metric}". Expected one of {valid_metrics()}')


def random(Y_mean: np.ndarray) -> np.ndarray:
    """Random acquistion score

    Parameters
    ----------
    Y_mean : np.ndarray
        an array of length equal to the number of random scores to generate.
        It is only used to determine the dimension of the output array, so the
        values contained in it are meaningless.

    Returns
    -------
    np.ndarray
        the random acquisition scores
    """
    return RG.random(len(Y_mean))


def threshold(Y_mean: np.ndarray, t: float) -> np.ndarray:
    """Random acquisition score [0, 1) if at or above threshold. Otherwise,
    return -1.

    Parameters
    ----------
    Y_mean : np.ndarray
    t : float
        the threshold value below which to assign acquisition scores as -1 and
        above or equal to which to assign random acquisition scores in the
        range (0, 1]

    Returns
    -------
    np.ndarray
        the random threshold acquisition scores
    """
    return np.where(Y_mean >= t, RG.random(Y_mean.shape), -1.0)


def greedy(Y_mean: np.ndarray) -> np.ndarray:
    """Greedy acquisition score

    Parameters
    ----------
    Y_mean : np.ndarray
        the mean predicted y values

    Returns
    -------
    np.ndarray
        the greedy acquisition scores
    """
    return Y_mean


def noisy(Y_mean: np.ndarray) -> np.ndarray:
    """Noisy greedy acquisition score

    Adds a random amount of noise to each predicted mean value. The noise is
    randomly sampled from a normal distribution centered at 0 with standard
    deviation equal to the standard deviation of the input predicted means.
    """
    sd = np.std(Y_mean)
    noise = RG.normal(scale=sd, size=len(Y_mean))
    return Y_mean + noise


def ucb(Y_mean: np.ndarray, Y_var: np.ndarray, beta: int = 2) -> np.ndarray:
    """Upper confidence bound acquisition score

    Parameters
    ----------
    Y_mean : np.ndarray
    Y_var : np.ndarray
        the variance of the mean predicted y values
    beta : int (Default = 2)
        the number of standard deviations to add to Y_mean

    Returns
    -------
    np.ndarray
        the upper confidence bound acquisition scores
    """
    return Y_mean + beta * np.sqrt(Y_var)


def lcb(Y_mean: np.ndarray, Y_var: np.ndarray, beta: int = 2) -> np.ndarray:
    """Lower confidence bound acquisition score

    Parameters
    ----------
    Y_mean : np.ndarray
    Y_var : np.ndarray
    beta : int (Default = 2)

    Returns
    -------
    np.ndarray
        the lower confidence bound acquisition scores
    """
    return Y_mean - beta * np.sqrt(Y_var)


def thompson(Y_mean: np.ndarray, Y_var: np.ndarray, stochastic: bool = False) -> np.ndarray:
    """Thompson acquisition score

    Parameters
    -----------
    Y_mean : np.ndarray
    Y_var : np.ndarray
    stochastic : bool
        is Y_mean generated stochastically?

    Returns
    -------
    np.ndarray
        the thompson acquisition scores
    """
    if stochastic:
        return Y_mean

    Y_sd = np.sqrt(Y_var)

    return RG.normal(Y_mean, Y_sd)


def ei(Y_mean: np.ndarray, Y_var: np.ndarray, current_max: float, xi: float = 0.01) -> np.ndarray:
    """Exected improvement acquisition score

    Parameters
    ----------
    Y_mean : np.ndarray
    Y_var : np.ndarray
    current_max : float
        the current maximum observed score
    xi : float (Default = 0.01)
        the amount by which to shift the improvement score

    Returns
    -------
    E_imp : np.ndarray
        the expected improvement acquisition scores
    """
    I = Y_mean - current_max + xi
    Y_sd = np.sqrt(Y_var)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = I / Y_sd
    E_imp = I * norm.cdf(Z) + Y_sd * norm.pdf(Z)

    # if the expected variance is 0, the expected improvement is the predicted improvement
    mask = Y_var == 0
    E_imp[mask] = I[mask]

    return E_imp


def pi(Y_mean: np.ndarray, Y_var: np.ndarray, current_max: float, xi: float = 0.01) -> np.ndarray:
    """Probability of improvement acquisition score

    Parameters
    ----------
    Y_mean : np.ndarray
    Y_var : np.ndarray
    current_max : float
    xi : float (Default = 0.01)

    Returns
    -------
    P_imp : np.ndarray
        the probability of improvement acquisition scores
    """
    I = Y_mean - current_max + xi
    with np.errstate(divide="ignore"):
        Z = I / np.sqrt(Y_var)
    P_imp = norm.cdf(Z)

    # if expected variance is 0, probability of improvement is 0 or 1 depending on whether the
    # predicted improvement is <= 0 or >0
    mask = Y_var == 0
    P_imp[mask] = np.where(I > 0, 1, 0)[mask]

    return P_imp


def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Calculate Tanimoto similarity between two fingerprints

    Parameters
    ----------
    fp1 : np.ndarray
        First fingerprint (binary vector)
    fp2 : np.ndarray
        Second fingerprint (binary vector)

    Returns
    -------
    float
        Tanimoto similarity in [0, 1]
    """
    intersection = np.sum(fp1 * fp2)
    union = np.sum(fp1) + np.sum(fp2) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def tanimoto_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Calculate Tanimoto distance (1 - similarity) between two fingerprints

    Parameters
    ----------
    fp1 : np.ndarray
        First fingerprint (binary vector)
    fp2 : np.ndarray
        Second fingerprint (binary vector)

    Returns
    -------
    float
        Tanimoto distance in [0, 1]
    """
    return 1.0 - tanimoto_similarity(fp1, fp2)


def batch_tanimoto_distance(fps: np.ndarray, selected_fps: np.ndarray) -> np.ndarray:
    """Calculate average Tanimoto distance from each fingerprint to a set of selected fingerprints

    Parameters
    ----------
    fps : np.ndarray
        Array of fingerprints (n_candidates x n_bits)
    selected_fps : np.ndarray
        Array of selected fingerprints (n_selected x n_bits)

    Returns
    -------
    np.ndarray
        Average Tanimoto distance for each candidate (n_candidates,)
    """
    if len(selected_fps) == 0:
        # If no molecules selected yet, return maximum diversity (all 1s)
        return np.ones(len(fps))

    # Calculate Tanimoto similarity matrix: (n_candidates x n_selected)
    # Using vectorized operations for efficiency
    intersection = fps @ selected_fps.T  # dot product gives intersection
    fps_sum = np.sum(fps, axis=1, keepdims=True)  # (n_candidates, 1)
    selected_sum = np.sum(selected_fps, axis=1)  # (n_selected,)
    union = fps_sum + selected_sum - intersection  # broadcasting

    # Avoid division by zero
    union = np.maximum(union, 1e-10)
    similarities = intersection / union

    # Convert to distances and take average
    distances = 1.0 - similarities
    avg_distances = np.mean(distances, axis=1)

    return avg_distances

def batch_min_tanimoto_distance(
    fps: np.ndarray, selected_fps: np.ndarray
) -> np.ndarray:
    """Calculate the MIN Tanimoto distance from each fingerprint to a set of
    selected fingerprints (MaxMin diversity).

    Unlike batch_tanimoto_distance which returns the AVERAGE distance,
    this returns the MINIMUM distance — i.e., the distance to the
    nearest neighbor in the selected set. This is the standard
    definition of diversity in MaxMin selection and is much more
    discriminative than average distance.

    Rationale
    ---------
    A candidate molecule's "novelty" should be determined by how far
    its CLOSEST already-selected analogue is, not by its average
    distance to all selected molecules. Average distance is dominated
    by the global Tanimoto distribution (~0.8 for random pairs in
    typical libraries) and quickly loses discriminative power once
    the selected set grows.

    Parameters
    ----------
    fps : np.ndarray
        Array of candidate fingerprints (n_candidates x n_bits)
    selected_fps : np.ndarray
        Array of already-selected fingerprints (n_selected x n_bits)

    Returns
    -------
    np.ndarray
        Min Tanimoto distance for each candidate (n_candidates,)
        Returns an array of ones if selected_fps is empty.
    """
    if len(selected_fps) == 0:
        # No molecules selected yet → maximum possible diversity for all
        return np.ones(len(fps))

    # Vectorized Tanimoto similarity: (n_candidates x n_selected)
    intersection = fps @ selected_fps.T
    fps_sum = np.sum(fps, axis=1, keepdims=True)        # (n_candidates, 1)
    selected_sum = np.sum(selected_fps, axis=1)         # (n_selected,)
    union = fps_sum + selected_sum - intersection       # broadcast
    union = np.maximum(union, 1e-10)                    # avoid /0
    similarities = intersection / union

    # Tanimoto distance = 1 - similarity, then take the MIN over the
    # selected set for each candidate (nearest-neighbor distance)
    distances = 1.0 - similarities
    min_distances = np.min(distances, axis=1)

    return min_distances

def md_ei(
    Y_mean: np.ndarray,
    Y_var: np.ndarray,
    current_max: float,
    xi: float = 0.01,
    fingerprints: Optional[np.ndarray] = None,
    selected_indices: Optional[np.ndarray] = None,
    alpha: float = 0.8,
) -> np.ndarray:
    """Molecular Diversity-aware Expected Improvement acquisition score

    Combines expected improvement with molecular diversity to encourage
    exploration of diverse chemical space while exploiting promising regions.

    MD-EI(x) = α * EI(x) + (1-α) * Diversity(x, selected)

    where:
    - EI(x): Expected Improvement
    - Diversity(x, selected): Average Tanimoto distance to selected molecules
    - α: Balance parameter (higher α = more exploitation)

    Parameters
    ----------
    Y_mean : np.ndarray
        Predicted mean values
    Y_var : np.ndarray
        Predicted variance values
    current_max : float
        Current maximum observed score
    xi : float (Default = 0.01)
        Exploration parameter for EI
    fingerprints : Optional[np.ndarray] (Default = None)
        Molecular fingerprints (n_candidates x n_bits)
        If None, falls back to standard EI
    selected_indices : Optional[np.ndarray] (Default = None)
        Indices of already selected molecules
        If None or empty, falls back to standard EI
    alpha : float (Default = 0.8)
        Balance parameter between exploitation (EI) and exploration (diversity)
        - α = 1.0: pure EI (no diversity)
        - α = 0.0: pure diversity (no EI)
        - α = 0.8: recommended default (80% EI, 20% diversity)

    Returns
    -------
    np.ndarray
        MD-EI acquisition scores

    References
    ----------
    Inspired by:
    - Reker & Schneider (2015). "Active-learning strategies in computer-assisted
      drug discovery." Drug Discovery Today.
    - Graff et al. (2021). "Accelerating high-throughput virtual screening through
      molecular pool-based active learning." Chemical Science.
    """
    # Calculate standard Expected Improvement
    E_imp = ei(Y_mean, Y_var, current_max, xi)

    # If no fingerprints or selected indices provided, return standard EI
    if fingerprints is None or selected_indices is None or len(selected_indices) == 0:
        return E_imp

    # Normalize EI to [0, 1] for combination with diversity
    E_imp_normalized = E_imp / (np.max(E_imp) + 1e-10)

    # Calculate diversity scores
    selected_fps = fingerprints[selected_indices]
    diversity_scores = batch_tanimoto_distance(fingerprints, selected_fps)

    # Combine EI and diversity
    md_ei_scores = alpha * E_imp_normalized + (1 - alpha) * diversity_scores

    return md_ei_scores

