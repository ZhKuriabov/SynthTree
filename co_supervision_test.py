#!/usr/bin/env python3
"""
automated_experiments.py
================================================
Run baseline vs. co-supervision-augmented students (CART, LogReg/LinReg, LRT)
on multiple tabular datasets while *repeating* the whole train/test procedure K times
with different random seeds, then reporting **mean ± SD** per setting.
"""
from __future__ import annotations
import argparse
import warnings
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from example_preprocessing import prep_data

from rpy2 import robjects as ro
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# Load Rforestry package
r('if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")')
r('library(devtools)')
r('devtools::load_all("../Rforestry")')
forestry = importr("Rforestry")

import numpy as np

def augment_cluster_data(X_cluster, model, num_aug=100, noise_std='auto', covariance_type='full'):
    """
    Generate synthetic samples around X_cluster and label using model.predict.

    Parameters:
        X_cluster : np.ndarray
            Data points in the cluster
        model : fitted model
            Used to predict labels for synthetic data
        num_aug : int
            Number of augmented points
        noise_std : float or 'auto'
            Multiplier for cluster spread. If 'auto', uses 1.0 * empirical covariance.
        covariance_type : 'full' or 'diag'
            Whether to use full covariance or just diagonal (per-feature variance)
    """
    mean = np.mean(X_cluster, axis=0)

    if covariance_type == 'full':
        cov = np.cov(X_cluster.T) + np.eye(X_cluster.shape[1]) * 1e-6
    elif covariance_type == 'diag':
        var = np.var(X_cluster, axis=0) + 1e-6
        cov = np.diag(var)
    elif covariance_type is None:
        cov = np.eye(X_cluster.shape[1])
    else:
        raise ValueError("covariance_type must be 'full' or 'diag'")

    if noise_std == 'auto':
        noise_std = 10

    samples = np.random.multivariate_normal(mean, noise_std * cov, size=num_aug)
    # ✅ Label synthetic samples using soft threshold if available
    if hasattr(model, 'predict_proba'):
        y_synthetic = (model.predict_proba(samples)[:, 1] > 0.5).astype(int)
    else:
        y_synthetic = model.predict(samples)

    unique, counts = np.unique(y_synthetic, return_counts=True)
    # print(f"Cluster augmentation: class distribution = {dict(zip(unique, counts))}")

    return samples, y_synthetic

class ClusteringManager:
    def __init__(self, max_k=15, min_k=2, method='silhouette'):
        self.max_k = max_k
        self.min_k = min_k
        self.method = method

    def find_best_k(self, X):
        scores = []
        for k in range(self.min_k, self.max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            scores.append((k, score))
        best_k = max(scores, key=lambda x: x[1])[0]
        return best_k, scores

    def cluster(self, X, k):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        labels = kmeans.fit_predict(X)
        return labels, kmeans.cluster_centers_


# RForestry wrapper classes
class _ForestryMixin(BaseEstimator):
    _ntree: int = 1

    def __init__(self):
        self.model_ = None

    def fit(self, X, y):
        with localconverter(default_converter + pandas2ri.converter):
            ro.globalenv["X"] = pd.DataFrame(X)
            ro.globalenv["y"] = pd.Series(y)
        ro.r(f"model <- forestry(x = X, y = y, ntree = {self._ntree})")
        self.model_ = ro.globalenv["model"]
        return self

    def _predict_r(self, newX):
        with localconverter(default_converter + pandas2ri.converter):
            ro.globalenv["X_new"] = pd.DataFrame(newX)
        preds = ro.r("predict(model, newdata = X_new, feature.new = X_new)")
        return np.asarray(preds, dtype=float)

class LRTClassifier(_ForestryMixin, ClassifierMixin):
    _ntree = 1
    def predict_proba(self, X):
        p = self._predict_r(X)
        return np.column_stack([1 - p, p])
    def predict(self, X):
        return (self._predict_r(X) > 0.5).astype(int)

class LRTRegressor(_ForestryMixin, RegressorMixin):
    _ntree = 1
    def predict(self, X):
        return self._predict_r(X)

class LRFClassifier(LRTClassifier): _ntree = 100
class LRFRegressor(LRTRegressor): _ntree = 100

# Metrics
rmse = lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)

# Interpretability metrics
from sklearn.tree import BaseDecisionTree

def interpretability_cart(tree_model: BaseDecisionTree) -> float:
    tree = tree_model.tree_
    is_leaf = (tree.children_left == -1)
    depths = np.zeros(tree.node_count, dtype=int)
    def compute_depths(node_id=0, depth=0):
        depths[node_id] = depth
        if tree.children_left[node_id] != -1:
            compute_depths(tree.children_left[node_id], depth + 1)
            compute_depths(tree.children_right[node_id], depth + 1)
    compute_depths()
    leaf_ids = np.where(is_leaf)[0]
    leaf_depths = depths[leaf_ids]
    leaf_samples = tree.n_node_samples[leaf_ids]
    N = leaf_samples.sum()
    return np.sum(leaf_samples * leaf_depths) / N

def interpretability_lrt(r_model) -> float:
    ro.globalenv["model"] = r_model
    interp = ro.r("get_leaf_info(model)")[1][0]
    # print("interp", interp)
    return float(interp)

# Grid search params
TEACHER_PARAM_GRIDS = {
    "RF": {"n_estimators": [100, 200], "max_depth": [None, 10]},
    "GB": {"n_estimators": [100], "learning_rate": [0.1, 0.05], "max_depth": [3, 5]},
    "MLP": {"hidden_layer_sizes": [(50,), (100,)], "alpha": [0.0001, 0.001]}
}

# Registries
TEACHERS = {
    "RF": (RandomForestClassifier(random_state=0), RandomForestRegressor(random_state=0)),
    "GB": (GradientBoostingClassifier(random_state=0), GradientBoostingRegressor(random_state=0)),
    "MLP": (MLPClassifier(max_iter=1000, random_state=0), MLPRegressor(max_iter=1000, random_state=0)),
    "LRF": (LRFClassifier(), LRFRegressor()),
}
CLASSIF_STUDENTS = {"CART": DecisionTreeClassifier(random_state=0), "LRT": LRTClassifier()}
REG_STUDENTS = {"CART": DecisionTreeRegressor(random_state=0), "LRT": LRTRegressor()}
CLASS_DATASETS = {"SKCM", "Road Safety", "Compas", "Upselling"}
REG_DATASETS = {"Cal Housing", "Bike Sharing", "Abalone", "Servo"}
ALL_DATASETS = sorted(CLASS_DATASETS | REG_DATASETS)

# Augment
def augment_dataset(X_train, y_train, teacher_model, *, max_k=30, min_k=2, num_aug=100, noise_std=0.1):
    mgr = ClusteringManager(max_k=max_k, min_k=min_k)
    best_k, _ = mgr.find_best_k(X_train)
    labels, _ = mgr.cluster(X_train, best_k)
    teacher_model.fit(X_train, y_train)
    X_aug_list, y_aug_list = [], []
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        Xa, ya = augment_cluster_data(X_train[idx], teacher_model, num_aug=num_aug, noise_std=noise_std, covariance_type="full")
        X_aug_list.append(Xa); y_aug_list.append(ya)
    X_aug = np.vstack(X_aug_list); y_aug = np.concatenate(y_aug_list)
    return np.vstack([X_train, X_aug]), np.concatenate([y_train, y_aug])

# One experiment

def run_one(data_file: str, teacher_name: str, seed: int):
    X_train, y_train, X_test, y_test = prep_data(data_file, random_state=seed)
    cls_t, reg_t = TEACHERS[teacher_name]
    is_classif = data_file in CLASS_DATASETS
    base_model = cls_t if is_classif else reg_t

    teacher_model = None
    if teacher_name in TEACHER_PARAM_GRIDS:
        grid = GridSearchCV(base_model, TEACHER_PARAM_GRIDS[teacher_name], cv=3,
                            scoring="roc_auc" if is_classif else "neg_root_mean_squared_error", n_jobs=-1)
        grid.fit(X_train, y_train)
        teacher_model = grid.best_estimator_

    elif teacher_name == "LRF":
        best_score = -np.inf if is_classif else np.inf
        best_model = None
        for ntree in [100, 200, 500]:
            model = LRFClassifier() if is_classif else LRFRegressor()
            model._ntree = ntree
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1] if is_classif else model.predict(X_test)
            score = roc_auc_score(y_test, y_pred) if is_classif else rmse(y_test, y_pred)
            if (is_classif and score > best_score) or (not is_classif and score < best_score):
                best_score = score
                best_model = model
        teacher_model = best_model

    else:
        base_model.fit(X_train, y_train)
        teacher_model = base_model

    X_aug, y_aug = augment_dataset(X_train, y_train, teacher_model)
    students = CLASSIF_STUDENTS if is_classif else REG_STUDENTS
    metric = lambda yt, yp: roc_auc_score(yt, yp[:, 1]) if is_classif else rmse(yt, yp)
    predictor = lambda m, X: m.predict_proba(X) if is_classif else m.predict(X)

    rows = []
    for phase, (Xt, yt) in {"orig": (X_train, y_train), "aug": (X_aug, y_aug)}.items():
        for s_name, model in students.items():
            model.fit(Xt, yt)
            score = metric(y_test, predictor(model, X_test))
            interp = interpretability_cart(model) if s_name == "CART" else (
                     interpretability_lrt(model.model_) if s_name == "LRT" else None)
            rows.append({"dataset": data_file, "teacher": teacher_name, "student": s_name,
                         "phase": phase, "score": score, "seed": seed, "interpretability": interp})
    return rows

# Main CLI
def main():
    p = argparse.ArgumentParser(description="Run SynthTree augmentation experiments with repeats")
    p.add_argument("--runs", "-r", type=int, default=5, help="Number of random splits (default=1)")
    p.add_argument("dataset", nargs="?", choices=ALL_DATASETS, help="Single dataset to run")
    p.add_argument("teacher", nargs="?", choices=list(TEACHERS), help="Single teacher to run")
    args = p.parse_args()

    datasets = [args.dataset] if args.dataset else ALL_DATASETS
    teachers = [args.teacher] if args.teacher else list(TEACHERS)

    rng = np.random.RandomState(0)
    all_rows = []
    for run in range(args.runs):
        base_seed = run
        print(f"=== Run {run + 1}/{args.runs} (base seed={base_seed}) ===")
        for ds in datasets:
            for t in teachers:
                print(f"→ {ds} | teacher={t}")
                success = False
                trial = 0
                while not success and trial < 10:
                    seed = base_seed * 100 + trial
                    try:
                        rows = run_one(ds, t, seed)
                        all_rows.extend(rows)
                        success = True
                    except Exception as e:
                        print(f"Failed on seed {seed}: {e} — retrying...")
                        trial += 1
                if not success:
                    print(f"Failed after 10 trials for {ds}, teacher={t}")

    df = pd.DataFrame(all_rows)
    df.to_csv(f"all_results_{datasets}.csv", index=False)
    summary = df.groupby(["dataset", "teacher", "student", "phase"]).agg({
        "score": ["mean", "std"],
        "interpretability": ["mean", "std"]
    }).reset_index()
    summary.columns = ['dataset','teacher','student','phase','score_mean','score_std','interp_mean','interp_std']
    summary["mean±std"] = summary.apply(lambda r: f"{r['score_mean']:.4f} ± {r['score_std']:.4f}", axis=1)
    summary["interp±std"] = summary.apply(lambda r: f"{r['interp_mean']:.2f} ± {r['interp_std']:.2f}", axis=1)
    # summary[['dataset','teacher','student','phase','mean±std','interp±std']].to_csv(f"all_results_summary_{datasets}.csv", index=False)
    summary.to_csv(f'summarized_results_{datasets}.csv', index=False)

    print("\nSummary (mean ± std):")
    print(summary[['dataset','teacher','student','phase','mean±std','interp±std']])
    # print(f"\nSaved: all_results_new{datasets}.csv, all_results_summary_{datasets}.csv")

    os.system('say "Experiment finished"')  # macOS TTS

if __name__ == "__main__":
    main()