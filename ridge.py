#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import random
import time
import warnings
from typing import Optional, Union, Callable, List, Tuple

from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import get_scorer, make_scorer
from sklearn.base import clone
from joblib import Parallel, delayed

random.seed(0)


# %%
def correlation_score(y_true, y_pred):
    # Compute correlation for each feature dimension (e.g., MEG source vertex)
    if y_true.ndim == 1:
        a = pearsonr(y_true, y_pred)[0]
        return a
    else:
        corrs = [
            pearsonr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])
        ]
        return np.nanmedian(corrs)


# Wrap in scikit-learn scorer format
correlation_scorer = make_scorer(correlation_score, greater_is_better=True)


class Ridge:
    """
    Supports two modes: n_perm=0 (no permutations) or n_perm=1000 (permutation testing)
    """

    def __init__(
        self,
        n_splits: int,
        alphas: np.ndarray = np.logspace(-3, 3, 5),
        scoring: Union[str, Callable] = correlation_scorer,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        n_pca: Optional[int] = 100,
        n_perm: int = 0,
    ):
        """
        Initialize OptimizedMDPC

        Parameters:
        -----------
        n_splits : int
            Number of cross-validation splits
        alphas : array-like
            Alpha values for Ridge regression
        scoring : str or callable
            Scoring function for cross-validation
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        random_state : int or None
            Random state for reproducibility
        n_pca : int or None
            Number of PCA components (None to skip PCA)
        n_perm : int
            Number of permutations (0 or 1000 only)
        """
        if n_perm not in [0, 1000]:
            raise ValueError("n_perm must be either 0 or 1000")

        self.n_splits = n_splits
        self.alphas = alphas
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_pca = n_pca
        self.n_perm = n_perm

    def _create_pipeline(self) -> Pipeline:
        """Create the model pipeline"""
        pipeline_steps = [("scaler", StandardScaler())]

        if self.n_pca:
            pipeline_steps.append(("pca", PCA(n_components=self.n_pca)))

        ridge_params = {"alphas": self.alphas, "scoring": self.scoring}

        # Add alpha_per_target for multi-target when using PCA
        if self.n_pca and self.y.ndim > 1:
            ridge_params["alpha_per_target"] = True

        pipeline_steps.append(("ridge", RidgeCV(**ridge_params)))

        return Pipeline(steps=pipeline_steps)

    def _single_fold_score(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        pipeline: Pipeline,
    ) -> float:
        """Compute score for a single fold"""
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline_clone = clone(pipeline)
        pipeline_clone.fit(x_train, y_train)
        scorer = get_scorer(self.scoring)
        return scorer(pipeline_clone, x_test, y_test)

    def _cross_validate_single_perm(
        self, perm_idx: int, fold_indices: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[float]:
        """Perform cross-validation for a single permutation"""
        # Set random seed for this permutation
        np.random.seed(self.random_state + perm_idx if self.random_state else perm_idx)

        # Create permuted data
        order = np.random.permutation(self.original_x.shape[0])
        x_perm = self.original_x[order]

        # Create pipeline
        pipeline = self._create_pipeline()

        scores = []
        for train_idx, test_idx in fold_indices:
            score = self._single_fold_score(
                train_idx, test_idx, x_perm, self.y, pipeline
            )
            scores.append(score)

        return scores

    def _run_no_permutation(self) -> List[float]:
        """Simple cross-validation without permutations"""
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        pipeline = self._create_pipeline()
        scores = []

        for train_idx, test_idx in kf.split(self.x):
            score = self._single_fold_score(
                train_idx, test_idx, self.x, self.y, pipeline
            )
            scores.append(score)

        return scores

    def _run_1000_permutations(self) -> np.ndarray:
        """Run 1000 permutations with parallel processing"""
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        # Pre-compute fold indices
        fold_indices = list(kf.split(self.original_x))

        # Run permutations in parallel
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            all_scores = Parallel(n_jobs=self.n_jobs, verbose=1)(
                delayed(self._cross_validate_single_perm)(perm_idx, fold_indices)
                for perm_idx in range(1000)
            )

        return np.array(all_scores).T  # Shape: (n_splits, 1000)

    def bv_linear(
        self, x: np.ndarray, y: np.ndarray, verbose: bool = False
    ) -> Union[List[float], np.ndarray]:
        """
        Main method for bivariate linear analysis

        Parameters:
        -----------
        x : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector/matrix
        verbose : bool
            Print timing information

        Returns:
        --------
        scores : List[float] or np.ndarray
            - If n_perm=0: List of CV scores (length n_splits)
            - If n_perm=1000: Array of shape (n_splits, 1000)
        """
        # Store data for this analysis
        self.x = np.array(x)
        self.y = np.array(y)
        self.original_x = self.x.copy()

        start_time = time.time()

        if self.n_perm == 0:
            if verbose:
                print("Running cross-validation without permutations...")
            scores = self._run_no_permutation()
        else:  # n_perm == 1000
            if verbose:
                print("Running 1000 permutations with parallel processing...")
            scores = self._run_1000_permutations()

        elapsed_time = time.time() - start_time

        if verbose:
            print(f"Completed in {elapsed_time:.2f} seconds")
            if self.n_perm == 1000:
                print(f"Average time per permutation: {elapsed_time/1000:.3f} seconds")

        return scores

    def compute_pvalue(self, perm_scores: np.ndarray, observed_score: float) -> float:
        """
        Compute p-value from 1000 permutation scores

        Parameters:
        -----------
        perm_scores : np.ndarray
            Permutation scores from bv_linear() with n_perm=1000
        observed_score : float
            The actual observed score (mean across CV folds)

        Returns:
        --------
        p_value : float
            Permutation p-value
        """
        if self.n_perm == 0:
            raise ValueError(
                "Cannot compute p-value without permutations (n_perm must be 1000)"
            )

        # Take mean across folds for each permutation
        mean_perm_scores = np.mean(perm_scores, axis=0)  # Shape: (1000,)

        # Count permutations with score >= observed score
        n_extreme = np.sum(mean_perm_scores >= observed_score)

        # Conservative p-value calculation
        p_value = (n_extreme + 1) / 1001

        return p_value
