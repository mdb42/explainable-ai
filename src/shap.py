"""
SHAP (SHapley Additive exPlanations) Implementation

This implementation follows the paper:
"A Unified Approach to Interpreting Model Predictions"
by Lundberg and Lee (2017)

1. Exact Shapley values (for small features)
2. Kernel SHAP (model-agnostic approximation)
3. Tree SHAP basics (for tree-based models)
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression
from typing import Callable, Optional, Union, List, Tuple
import warnings
from abc import ABC, abstractmethod


class SHAPExplainer(ABC):
    """Abstract base class for SHAP explainers."""
    
    @abstractmethod
    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Calculate SHAP values for given instances."""
        pass


class ExactSHAPExplainer(SHAPExplainer):
    """
    Exact SHAP value computation using the original Shapley value formula.
    
    Warning: This is computationally expensive (O(2^n_features)).
    Only suitable for small numbers of features (< 15).
    """
    
    def __init__(
        self,
        predict_fn: Callable,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize Exact SHAP explainer.
        
        Args:
            predict_fn: Function that takes data and returns predictions
            data: Background dataset for computing expectations
            feature_names: Names of features
        """
        self.predict_fn = predict_fn
        self.data = data
        self.n_features = data.shape[1]
        self.feature_names = feature_names or [f'Feature {i}' for i in range(self.n_features)]
        
        # Precompute background expectation
        self.base_value = self._compute_base_value()
        
    def _compute_base_value(self) -> float:
        """Compute expected model output over background data."""
        # TODO: Implement computation of E[f(x)]
        # This is the expected prediction over the background dataset
        raise NotImplementedError("Implement base value computation")
        
    def _generate_coalitions(self, n_features: int) -> List[Tuple[int, ...]]:
        """
        Generate all possible coalitions (subsets) of features.
        
        Args:
            n_features: Number of features
            
        Returns:
            List of all possible feature coalitions
        """
        # TODO: Generate all possible subsets of features
        # Hint: Use itertools.combinations
        raise NotImplementedError("Implement coalition generation")
        
    def _compute_marginal_contribution(
        self,
        instance: np.ndarray,
        feature_idx: int,
        coalition: Tuple[int, ...],
        n_samples: int = 100
    ) -> float:
        """
        Compute marginal contribution of a feature to a coalition.
        
        Args:
            instance: Instance to explain
            feature_idx: Index of feature to compute contribution for
            coalition: Current coalition of features (indices)
            n_samples: Number of samples for Monte Carlo approximation
            
        Returns:
            Marginal contribution of the feature
        """
        # TODO: Implement marginal contribution computation
        # 1. Create coalition with feature: coalition ∪ {feature_idx}
        # 2. Compute f(coalition with feature) - f(coalition without feature)
        # 3. Use Monte Carlo sampling from background data for missing features
        
        raise NotImplementedError("Implement marginal contribution")
        
    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute exact SHAP values using Shapley formula.
        
        Args:
            X: Instances to explain (n_instances, n_features)
            
        Returns:
            SHAP values (n_instances, n_features)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        n_instances = X.shape[0]
        shap_values = np.zeros((n_instances, self.n_features))
        
        # Warn if too many features
        if self.n_features > 15:
            warnings.warn(
                f"Computing exact SHAP values for {self.n_features} features "
                "will be very slow. Consider using KernelSHAP instead."
            )
        
        for i, instance in enumerate(X):
            # TODO: Implement exact Shapley value computation
            # For each feature j:
            #   1. Generate all coalitions not containing j
            #   2. For each coalition S:
            #      a. Compute weight: |S|!(M-|S|-1)!/M!
            #      b. Compute marginal contribution: v(S ∪ {j}) - v(S)
            #   3. Sum weighted marginal contributions
            
            raise NotImplementedError("Implement exact SHAP computation")
            
        return shap_values


class KernelSHAPExplainer(SHAPExplainer):
    """
    Kernel SHAP: Linear LIME + Shapley kernel weights = SHAP values
    
    This is a model-agnostic approximation that works for any model.
    """
    
    def __init__(
        self,
        predict_fn: Callable,
        data: np.ndarray,
        link: str = "identity",
        feature_names: Optional[List[str]] = None,
        n_samples: int = 2048,
        random_state: Optional[int] = None
    ):
        """
        Initialize Kernel SHAP explainer.
        
        Args:
            predict_fn: Function that takes data and returns predictions
            data: Background dataset (or summary statistics)
            link: Link function ('identity' or 'logit')
            feature_names: Names of features
            n_samples: Number of samples for approximation
            random_state: Random seed
        """
        self.predict_fn = predict_fn
        self.data = data
        self.link = link
        self.n_features = data.shape[1]
        self.feature_names = feature_names or [f'Feature {i}' for i in range(self.n_features)]
        self.n_samples = n_samples
        self.rng = np.random.RandomState(random_state)
        
        # Compute background statistics
        self._prepare_background_data()
        
    def _prepare_background_data(self):
        """Prepare background data statistics."""
        # TODO: Implement background data preparation
        # 1. If data is large, consider using kmeans to summarize
        # 2. Compute base_value: expected prediction over background
        # 3. Store background samples for later use
        
        raise NotImplementedError("Implement background data preparation")
        
    def _generate_coalition_samples(
        self,
        instance: np.ndarray,
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate coalition samples for Kernel SHAP.
        
        Args:
            instance: Instance to explain
            n_samples: Number of samples to generate
            
        Returns:
            masks: Binary coalition matrix (n_samples, n_features)
            weights: Shapley kernel weights for each coalition
        """
        # TODO: Implement coalition sampling
        # 1. Generate random binary coalitions
        # 2. Ensure to include empty and full coalitions
        # 3. Compute Shapley kernel weights for each coalition
        #    Weight formula: (M-1) / (binom(M, |S|) * |S| * (M - |S|))
        #    where M = n_features, |S| = coalition size
        
        raise NotImplementedError("Implement coalition sampling")
        
    def _compute_shapley_kernel_weight(
        self,
        coalition_size: int,
        n_features: int
    ) -> float:
        """
        Compute Shapley kernel weight for a coalition.
        
        Args:
            coalition_size: Number of features in coalition
            n_features: Total number of features
            
        Returns:
            Shapley kernel weight
        """
        # TODO: Implement Shapley kernel weight formula
        # Handle edge cases: coalition_size = 0 or n_features
        
        raise NotImplementedError("Implement Shapley kernel weight")
        
    def _mask_to_data(
        self,
        instance: np.ndarray,
        mask: np.ndarray,
        n_samples: int = 1
    ) -> np.ndarray:
        """
        Convert binary mask to actual data instances.
        
        Args:
            instance: Original instance
            mask: Binary mask indicating which features to keep
            n_samples: Number of samples to generate
            
        Returns:
            Data instances with masked features replaced by background
        """
        # TODO: Implement mask to data conversion
        # Where mask = 1: keep original feature value
        # Where mask = 0: replace with background samples
        
        raise NotImplementedError("Implement mask to data conversion")
        
    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values using Kernel SHAP approximation.
        
        Args:
            X: Instances to explain
            
        Returns:
            SHAP values
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        n_instances = X.shape[0]
        shap_values = np.zeros((n_instances, self.n_features))
        
        for i, instance in enumerate(X):
            # TODO: Implement Kernel SHAP algorithm
            # 1. Generate coalition samples and weights
            # 2. For each coalition:
            #    a. Convert mask to actual data
            #    b. Get model predictions
            # 3. Fit weighted linear regression
            # 4. Extract coefficients as SHAP values
            
            raise NotImplementedError("Implement Kernel SHAP")
            
        return shap_values


class TreeSHAPExplainer(SHAPExplainer):
    """
    Tree SHAP for tree-based models (simplified version).
    
    This is a fast, exact algorithm for tree ensembles.
    We'll implement a basic version for a single decision tree.
    """
    
    def __init__(
        self,
        tree_model,
        data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize Tree SHAP explainer.
        
        Args:
            tree_model: Scikit-learn tree model
            data: Background data (optional for tree models)
            feature_names: Names of features
        """
        self.model = tree_model
        self.tree = tree_model.tree_  # Access internal tree structure
        self.n_features = tree_model.n_features_in_
        self.feature_names = feature_names or [f'Feature {i}' for i in range(self.n_features)]
        
        # For tree models, we can compute exact expectations
        self._compute_tree_statistics()
        
    def _compute_tree_statistics(self):
        """Compute tree statistics needed for SHAP values."""
        # TODO: Implement tree statistics computation
        # 1. Compute leaf weights (prediction values)
        # 2. Compute node sample counts
        # 3. Compute path statistics
        
        raise NotImplementedError("Implement tree statistics")
        
    def _tree_path(self, instance: np.ndarray) -> List[int]:
        """
        Find path through tree for given instance.
        
        Args:
            instance: Input instance
            
        Returns:
            List of node indices in path from root to leaf
        """
        # TODO: Implement path finding
        # Follow decision path from root to leaf
        
        raise NotImplementedError("Implement tree path finding")
        
    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for tree model.
        
        This is a simplified version of the full TreeSHAP algorithm.
        """
        # TODO: Implement simplified Tree SHAP
        # For each instance:
        # 1. Find path through tree
        # 2. For each feature:
        #    a. Compute contribution along path
        #    b. Account for feature interactions in tree
        
        raise NotImplementedError("Implement Tree SHAP")


class SHAPExplanation:
    """Container for SHAP explanation results."""
    
    def __init__(
        self,
        values: np.ndarray,
        base_values: Union[float, np.ndarray],
        data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize SHAP explanation.
        
        Args:
            values: SHAP values
            base_values: Base value(s) (expected model output)
            data: Original instance data
            feature_names: Feature names
        """
        self.values = values
        self.base_values = base_values
        self.data = data
        self.feature_names = feature_names
        
    def verify_additivity(self, predict_fn: Callable, tolerance: float = 1e-5) -> bool:
        """
        Verify SHAP additivity property: f(x) = base_value + sum(shap_values).
        
        Args:
            predict_fn: Model prediction function
            tolerance: Numerical tolerance
            
        Returns:
            True if additivity holds
        """
        # TODO: Implement additivity check
        # sum(shap_values) + base_value should equal f(x)
        
        raise NotImplementedError("Implement additivity verification")
        
    def plot_waterfall(self, instance_idx: int = 0, max_features: int = 10):
        """
        Create waterfall plot for single prediction.
        
        Args:
            instance_idx: Index of instance to plot
            max_features: Maximum features to show
        """
        import matplotlib.pyplot as plt
        
        # TODO: Implement waterfall plot
        # 1. Sort features by absolute SHAP value
        # 2. Show cumulative sum from base value to prediction
        # 3. Color positive/negative contributions differently
        
        raise NotImplementedError("Implement waterfall plot")
        
    def plot_summary(self, plot_type: str = "dot"):
        """
        Create summary plot of SHAP values.
        
        Args:
            plot_type: 'dot' or 'bar'
        """
        import matplotlib.pyplot as plt
        
        # TODO: Implement summary plots
        # For 'dot': scatter plot showing distribution
        # For 'bar': mean absolute SHAP values
        
        raise NotImplementedError("Implement summary plot")


# Utility functions
def verify_shapley_properties(
    shap_values: np.ndarray,
    predict_fn: Callable,
    instance: np.ndarray,
    base_value: float,
    tolerance: float = 1e-5
) -> dict:
    """
    Verify that computed values satisfy Shapley value properties.
    
    Returns dict with verification results.
    """
    results = {}
    
    # TODO: Implement property verification
    # 1. Local accuracy (additivity)
    # 2. Missingness (zero contribution for constant features)
    # 3. Consistency (harder to verify, optional)
    
    raise NotImplementedError("Implement property verification")


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 1000, 5
    X = np.random.randn(n_samples, n_features)
    
    # Create ground truth linear model
    true_weights = np.array([2.0, -1.5, 0.5, 0.0, -0.8])
    y = X @ true_weights + 0.1 * np.random.randn(n_samples)
    
    # Train a model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    
    # Use linear model for easier verification
    model = LinearRegression()
    model.fit(X, y)
    
    print("True weights:", true_weights)
    print("Learned weights:", model.coef_)
    
    # Test instance
    test_instance = X[0]
    
    # For linear models, SHAP values should equal: feature_value * coefficient
    expected_shap = test_instance * model.coef_
    print("\nExpected SHAP values (for linear model):", expected_shap)
    
    try:
        # Test Exact SHAP (only for very few features)
        exact_explainer = ExactSHAPExplainer(
            predict_fn=model.predict,
            data=X[:100],  # Use subset for speed
            feature_names=[f'x{i}' for i in range(n_features)]
        )
        exact_shap = exact_explainer.shap_values(test_instance)
        print("\nExact SHAP values:", exact_shap)
        
    except NotImplementedError as e:
        print(f"\nExact SHAP not implemented: {e}")
        
    try:
        # Test Kernel SHAP
        kernel_explainer = KernelSHAPExplainer(
            predict_fn=model.predict,
            data=X[:100],
            n_samples=1000,
            random_state=42
        )
        kernel_shap = kernel_explainer.shap_values(test_instance)
        print("\nKernel SHAP values:", kernel_shap)
        
    except NotImplementedError as e:
        print(f"Kernel SHAP not implemented: {e}")
        
    # Additional test with tree model
    from sklearn.tree import DecisionTreeRegressor
    tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree_model.fit(X, y)
    
    try:
        tree_explainer = TreeSHAPExplainer(
            tree_model=tree_model,
            data=X[:100]
        )
        tree_shap = tree_explainer.shap_values(test_instance)
        print("\nTree SHAP values:", tree_shap)
        
    except NotImplementedError as e:
        print(f"Tree SHAP not implemented: {e}")