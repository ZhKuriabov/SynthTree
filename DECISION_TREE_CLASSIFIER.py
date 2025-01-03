import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score

def rmse(y_true,y_pred):
    return mean_squared_error(y_true,y_pred,squared=False)

class DistanceDecisionTree:
    def __init__(self, max_depth=None, classification=False):
        self.max_depth = max_depth
        self.tree = None
        self.final_depth = None  # Add a new attribute to store the final depth
        self.classification = classification

    def fit(self, location_matrix, distance_matrix, depth=0, parent=None):
        self.tree, self.final_depth = self._build_tree(location_matrix, distance_matrix, depth, parent)

    def _build_tree(self, location_matrix, distance_matrix, depth, parent=None):
        if depth == self.max_depth:
            # Create a leaf node and fit a logistic regression model later
            return {'leaf': True, 'data_indices': None, 'accuracy': None, 'location_matrix': location_matrix, 'parent': parent}, depth

        # Find the best split based on minimum average pairwise distances
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
        
#         if best_split is not None:
#             left_child, left_depth = self._build_tree(*best_split['left'], depth + 1, parent={'feature_index': best_split['feature_index'], 'threshold': best_split['threshold'], 'left': None, 'right': None})
#             right_child, right_depth = self._build_tree(*best_split['right'], depth + 1, parent={'feature_index': best_split['feature_index'], 'threshold': best_split['threshold'], 'left': None, 'right': None})
            
#             current_depth = max(left_depth, right_depth)
#             return {'feature_index': best_split['feature_index'],
#                     'threshold': best_split['threshold'],
#                     'left': left_child,
#                     'right': right_child,
#                     'parent': parent}, current_depth
#         else:
#             # Create a leaf node if no split is found
#             return {'leaf': True, 'data_indices': None, 'accuracy': None, 'location_matrix': location_matrix, 'parent': parent}, depth

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
                    # Handle the case where either side is empty
                    left_avg_distance = right_avg_distance = float('inf')
                    
        return best_split

    def fit_leaf_logistic_models_cluster_based(self, y, location_matrix, augmented_data, a):
        self._fit_leaf_logistic_models_cluster_based(self.tree, y, location_matrix, augmented_data, a)
        
    def _fit_leaf_logistic_models_cluster_based(self, node, y, location_matrix, augmented_data, a):
        node_clf = []
        if 'leaf' in node:
#             print(node['location_matrix'])
            for i in np.arange(location_matrix.shape[0]):
                for j in node['location_matrix']:
#                     if j in location_matrix[i]:
                    if np.allclose(j, location_matrix[i]):
                        node_clf.append(i)
            
#             print("node_clf", node_clf)

            data = [augmented_data[key] for key in node_clf]
            data_y = [y[key] for key in node_clf]
            
            X = pd.DataFrame([item for sublist in [arr.tolist() for arr in data] for item in sublist])
            response = pd.DataFrame([item for sublist in [arr.tolist() for arr in data_y] for item in sublist])
            
#             print("X:", X)
    
            if len(np.unique(response)) == 1:              
                if self.classification == False:
                    model = LinearRegression()
                    model.fit(X, response)
                else:
                    model = LogisticRegression(solver='saga', penalty='l1')
                model.classes_ = np.unique(response)
                model.coef_ = np.zeros((1, location_matrix.shape[1]))
                model.intercept_ = 0.0
            else:
                if self.classification == False:
#                     model = LinearRegression()
                    model = sm.OLS(response, X).fit(method='pinv')
                else:
#                     model = LogisticRegression(solver='saga', max_iter=1000)
                    
                    model = sm.Logit(response, X).fit_regularized(method='l1', alpha=a, maxiter=1000)
#                 model.fit(X, response)
#                 print("model_coef:", model.coef_)

            node['model'] = model
#             node['accuracy'] = model.score(X, response)
            if self.classification:
                node['accuracy'] = accuracy_score(model.predict(X), y)
            else:
                node['accuracy'] = rmse(model.predict(X), y)
        else:
            self._fit_leaf_logistic_models_cluster_based(node['left'], y, location_matrix, augmented_data, a)
            self._fit_leaf_logistic_models_cluster_based(node['right'], y, location_matrix, augmented_data, a)
        
        
    def fit_leaf_logistic_models(self, X, y, X_test=None, y_test=None):
        self._fit_leaf_logistic_models(self.tree, X, y)
        
    def _fit_leaf_logistic_models(self, node, X, y):
        if 'leaf' in node:
#             print("Data in leaf to fit LR:", np.arange(X.shape[0]))
            node['data_indices'] = np.arange(X.shape[0])
            if node['data_indices'].size == 0:
                # Handle the case where there's no data for this leaf
#                 node['data_indices'] = node['parent']['data_indices']
#                 node = node['parent']
#                 node['left'] = None
#                 node['right'] = None
#                 node['leaf'] = True
                node['model'] = None
                node['accuracy'] = None
            else:
                data_indices = node['data_indices']

                unique_classes, counts = np.unique(y[data_indices], return_counts=True)

                if len(unique_classes) == 1:
                    if self.classification == False:
                    # If there's only one class, create a model that predicts this class
                        model = LinearRegression()
                        model.fit(X[data_indices], y[data_indices])
                    else:
                        model = LogisticRegression(solver='saga', penalty='l1')
                    model.classes_ = unique_classes
                    model.coef_ = np.zeros((1, X.shape[1]))
                    model.intercept_ = 0.0
                else:
                    if self.classification == False:
                    # If there's only one class, create a model that predicts this class
#                         model = LinearRegression()
#                         model = sm.OLS(y[data_indices], X[data_indices]).fit_regularized(method='elastic_net', L1_wt=1, alpha=0, maxiter=1000)
                        model = sm.OLS(y[data_indices], X[data_indices]).fit()
#                         model.fit(X[data_indices], y[data_indices])
                    else:
                        
#                         model = LogisticRegression(solver='saga', penalty='l1')
                        model = sm.Logit(y, X).fit_regularized(method='l1', alpha=0.1, maxiter=1000)
#                         model.fit(X[data_indices], y[data_indices])

                node['model'] = model
    #                 node['accuracy'] = model.score(X[data_indices], y[data_indices])
                if self.classification:
                    node['accuracy'] = accuracy_score(np.round(model.predict(X)), y)
                else:
                    node['accuracy'] = rmse(model.predict(X), y)
        else:
            left_indices = X[:, node['feature_index']] <= node['threshold']
            right_indices = ~left_indices
#             left_indices_test = X_test[:, node['feature_index']] <= node['threshold']
#             right_indices_test = ~left_indices_test
            self._fit_leaf_logistic_models(node['left'], X[left_indices], y[left_indices])
            self._fit_leaf_logistic_models(node['right'], X[right_indices], y[right_indices])

    def predict(self, X, y):
        predictions = [self._predict_instance(x, self.tree, y[i]) for i, x in enumerate(X)]
#         print(predictions)
        return predictions

    def _predict_instance(self, x, node, y):
        if 'leaf' in node:
            if node['model']:
#             return node['model'].predict([x])
#             print("Prediction is happening:", len(x), type(x), x.shape)
#             st_x = sm.add_constant(x.reshape(1, -1), has_constant='add')
                return node['model'].predict(x.reshape(1, -1))
            elif node['parent']['left'].get('model', None) != None or node['parent']['right'].get('model', None) != None:
                # Current leaf has no model, try the other leaf in the same parent
                parent = node['parent']
                sibling_leaf = parent['left'] if node is parent['right'] else parent['right']

                # Recursively check the sibling leaf
                return self._predict_instance(x, sibling_leaf, y)
            else:
                parent = node['parent']
                # If both leaves and parent have no models, try the sibling node of the parent
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
#                 print(indent + node['location_matrix'])
                print(indent + f"Leaf: Accuracy Test= {node['accuracy']:.4f}")
                print(indent + "Model:", node['model'])
            else:
                print(indent + "Leaf: No data for this leaf.")
        else:
            print(indent + f"Feature {node['feature_index']} <= {node['threshold']}")
            print(indent + "Left:")
            self.print_tree(node['left'], indent + "  ")
            print(indent + "Right:")
            self.print_tree(node['right'], indent + "  ")