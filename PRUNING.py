import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score
import copy
from joblib import Parallel, delayed

def rmse(y_true,y_pred):
    return mean_squared_error(y_true,y_pred,squared=False)

class DistanceDecisionTree:
    def __init__(self, max_depth=None, classification=False):
        self.max_depth = max_depth
        self.tree = None
        self.final_depth = None
        self.classification = classification
        
    def copy(self):
        new_instance = DistanceDecisionTree(self.max_depth, self.classification)
        new_instance.tree = copy.deepcopy(self.tree)
        new_instance.final_depth = self.final_depth
        return new_instance

    def fit(self, location_matrix, distance_matrix, depth=0, parent=None):
        self.tree, self.final_depth = self._build_tree(location_matrix, distance_matrix, depth, parent)

    def _build_tree(self, location_matrix, distance_matrix, depth, parent=None):
        if depth == self.max_depth:
            return {'leaf': True, 'data_indices': None, 'accuracy': None, 'location_matrix': location_matrix, 'parent': parent}, depth

        best_split = self._find_best_split(location_matrix, distance_matrix)

        if best_split is not None:
            parent_node = {'feature_index': best_split['feature_index'], 'threshold': best_split['threshold'], 'left': None, 'right': None, 'parent': parent}

            left_child, left_depth = self._build_tree(*best_split['left'], depth + 1, parent=parent_node)
            parent_node['left'] = left_child

            right_child, right_depth = self._build_tree(*best_split['right'], depth + 1, parent=parent_node)
            parent_node['right'] = right_child

            current_depth = max(left_depth, right_depth)
            
            return parent_node, current_depth
        
        else:
            return {'leaf': True, 'data_indices': None, 'accuracy': None, 'location_matrix': location_matrix, 'parent': parent}, depth

    def _find_best_split(self, location_matrix, distance_matrix):
        best_split = None
        parent_avg_distance = sum(np.mean(distance_matrix, axis=1))/2
        min_avg_distance = parent_avg_distance
        num_samples = location_matrix.shape[0]

        for feature_index in range(location_matrix.shape[1]):
            thresholds = np.unique(location_matrix[:, feature_index])

            for threshold in thresholds:
                left_mask = location_matrix[:, feature_index] <= threshold
                right_mask = ~left_mask

                if np.any(left_mask) and np.any(right_mask):
                    left_avg_distance = sum(np.mean(distance_matrix[left_mask][:, left_mask], axis=1))/2
                    right_avg_distance = sum(np.mean(distance_matrix[right_mask][:, right_mask], axis=1))/2
                    avg_distance = (len(location_matrix[left_mask]) * left_avg_distance +
                                    len(location_matrix[right_mask]) * right_avg_distance) / len(location_matrix)


                    if avg_distance < min_avg_distance and len(distance_matrix) > 1:
                        min_avg_distance = avg_distance
                        left_child = (location_matrix[left_mask], distance_matrix[left_mask][:, left_mask])
                        right_child = (location_matrix[right_mask], distance_matrix[right_mask][:, right_mask])
                        parent = (location_matrix, distance_matrix[:][:, :])
                        best_split = {'feature_index': feature_index,
                                      'threshold': threshold,
                                      'left': left_child,
                                      'right': right_child}
                else:
                    left_avg_distance = right_avg_distance = float('inf')

        return best_split
        
    def fit_leaf_logistic_models(self, X, y, X_test=None, y_test=None):
        self._fit_leaf_logistic_models(self.tree, X, y)
        
    def _fit_leaf_logistic_models(self, node, X, y):
        node['X'] = X
        node['y'] = y
        if 'leaf' in node:
            node['data_indices'] = np.arange(X.shape[0])
            if node['data_indices'].size == 0:
                node['model'] = None
                node['accuracy'] = None
                node['R_node'] = 10**10
            else:
                data_indices = node['data_indices']

                unique_classes, counts = np.unique(y[data_indices], return_counts=True)

                if len(unique_classes) == 1:
                    if self.classification == False:
                        model = LinearRegression()
                        model.fit(X[data_indices], y[data_indices])
                    elif self.classification == True:
                        model = LogisticRegression(solver='saga', penalty='l1')
                    model.classes_ = unique_classes
                    model.coef_ = np.zeros((1, X.shape[1]))
                    model.intercept_ = 0.0
                else:
                    if self.classification == False:
                        model = sm.OLS(y[data_indices], X[data_indices]).fit(disp=0)
                    elif self.classification == True:
                        model = sm.Logit(y[data_indices], X[data_indices]).fit_regularized(method='l1', alpha=0.1, maxiter=10000)
                node['model'] = model

                if self.classification == True:
                    node['R_node'] = np.sum(y != np.round(model.predict(X)))
                elif self.classification == False:
                    node['R_node'] = mean_squared_error(y, model.predict(X))
    
                if self.classification == True:
                    node['accuracy'] = accuracy_score(np.round(model.predict(X)), y)
                elif self.classification == False:
                    node['accuracy'] = rmse(model.predict(X), y)
        else:
            node['data_indices'] = np.arange(X.shape[0])
            if node['data_indices'].size == 0:
                node['R_node'] = 10^10
                node['model'] = None
                node['accuracy'] = None
            else:
                data_indices = node['data_indices']

                unique_classes, counts = np.unique(y[data_indices], return_counts=True)

                if len(unique_classes) == 1:
                    if self.classification == False:
                        model = LinearRegression()
                        model.fit(X[data_indices], y[data_indices])
                    elif self.classification == True:
                        model = LogisticRegression(solver='saga', penalty='l1')
                    model.classes_ = unique_classes
                    model.coef_ = np.zeros((1, X.shape[1]))
                    model.intercept_ = 0.0
                else:
                    if self.classification == False:
                        model = sm.OLS(y[data_indices], X[data_indices]).fit(disp=0)
                    elif self.classification == True:
                        model = sm.Logit(y[data_indices], X[data_indices]).fit_regularized(method='l1', alpha=0.1, maxiter=10000)
                node['model'] = model
                
                if self.classification == True:
                    node['R_node'] = np.sum(y != np.round(model.predict(X)))
                elif self.classification == False:
                    node['R_node'] = len(X) * mean_squared_error(y, model.predict(X))
                
                if self.classification == True:
                    node['accuracy'] = accuracy_score(np.round(model.predict(X)), y)
                elif self.classification == False:
                    node['accuracy'] = rmse(model.predict(X), y)
                    
            left_indices = X[:, node['feature_index']] <= node['threshold']
            right_indices = ~left_indices
            self._fit_leaf_logistic_models(node['left'], X[left_indices], y[left_indices])
            self._fit_leaf_logistic_models(node['right'], X[right_indices], y[right_indices])

    def predict(self, X, y):
        predictions = [self._predict_instance(x, self.tree, y[i]) for i, x in enumerate(X)]
        return predictions

    def _predict_instance(self, x, node, y):
        if 'leaf' in node:
            if node['model']:
                return node['model'].predict(x.reshape(1, -1))
            elif node['parent']['left'].get('model', None) != None or node['parent']['right'].get('model', None) != None:
                parent = node['parent']
                sibling_leaf = parent['left'] if node is parent['right'] else parent['right']

                return self._predict_instance(x, sibling_leaf, y)
            else:
                parent = node['parent']
                parent_sibling_node = parent['parent']['left'] if parent is parent['parent']['right'] else parent['parent']['right']
                return self._predict_instance(x, parent_sibling_node, y)
        
        else:
            if x[node['feature_index']] <= node['threshold']:
                return self._predict_instance(x, node['left'], y)
            else:
                return self._predict_instance(x, node['right'], y)

    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.tree

        if 'leaf' in node:
            if node['model']:
                print(indent + f"Leaf: Accuracy Test= {node['accuracy']:.4f}")
                print(indent + "Model:", node['model'], indent + 'R_node:', node['R_node'])
            else:
                print(indent + "Leaf: No data for this leaf.")
        else:
            print(indent + f"Feature {node['feature_index']} <= {node['threshold']}" + indent + "Model:", node['model'], indent + 'R_node:', node['R_node'])
            print(indent + "Left:")
            self.print_tree(node['left'], indent + "  ")
            print(indent + "Right:")
            self.print_tree(node['right'], indent + "  ")

    def interp_info(self):
        """
        Compute interpretability score and return leaf-level info for a DistanceDecisionTree.

        Returns
        -------
        leaf_info : pd.DataFrame
            Columns: ['leaf_id', 'depth', 'num_observations', 'k_l', 'p']
        interpretability_raw : float
            Weighted average path length
        """
        leaf_nodes = []

        def traverse(node, depth=0):
            if 'leaf' in node and node['leaf']:
                n_obs = node['X'].shape[0] if 'X' in node and node['X'] is not None else 0
                leaf_nodes.append((node, depth, n_obs))
            else:
                if node.get('left') is not None:
                    traverse(node['left'], depth + 1)
                if node.get('right') is not None:
                    traverse(node['right'], depth + 1)

        traverse(self.tree)

        rows = []
        p = None
        for leaf_id, (node, depth, n_obs) in enumerate(leaf_nodes):
            model = node.get('model', None)
            if model is not None:
                # grab coefficients (sklearn vs. statsmodels)
                try:
                    coef = model.coef_.ravel()
                except AttributeError:
                    # statsmodels: first param is intercept
                    params = np.asarray(model.params)
                    coef = params[1:]
                k_l = int(np.count_nonzero(coef))
                if p is None:
                    p = coef.size
            else:
                k_l = 0
                # if no model ever set p, fall back to number of features
                if p is None and 'X' in node and node['X'] is not None:
                    p = node['X'].shape[1]
            rows.append((leaf_id, depth, n_obs, k_l, p))

        df = pd.DataFrame(rows, columns=["leaf_id", "depth", "num_observations", "k_l", "p"])
        N = df["num_observations"].sum()

        if N > 0:
            weighted_depth = (df["depth"] * df["num_observations"]).sum() / N
            weighted_complexity = (df["num_observations"] * (1 + df["k_l"])).sum() / N
            interpretability_raw = 0.5 * weighted_depth + 0.5 * weighted_complexity
        else:
            interpretability_raw = np.nan

        return df, interpretability_raw