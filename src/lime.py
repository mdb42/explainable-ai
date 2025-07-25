"""
LIME (Local Interpretable Model-agnostic Explanations) Implementation

This implementation will follow the paper:
"Why Should I Trust You?" Explaining the Predictions of Any Classifier
by Ribeiro, Singh, and Guestrin (2016)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import euclidean_distances
from typing import Callable, Optional, Union, Tuple
import warnings


class LimeTabularExplainer:
    """
    LIME explainer for tabular data.
    
    This implementation focuses on continuous features for simplicity,
    but can be extended to handle categorical features.
    """
    
    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: Optional[list] = None,
        class_names: Optional[list] = None,
        mode: str = 'classification',
        discretize_continuous: bool = True,
        kernel_width: Optional[float] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize LIME explainer.
        
        Args:
            training_data: numpy array of training data used to generate statistics
            feature_names: list of feature names
            class_names: list of class names (for classification)
            mode: 'classification' or 'regression'
            discretize_continuous: whether to discretize continuous features
            kernel_width: kernel width for exponential kernel (default: sqrt(n_features) * 0.75)
            random_state: random seed for reproducibility
        """
        self.training_data = training_data
        self.n_features = training_data.shape[1]
        
        self.feature_names = feature_names or [f'feature_{i}' for i in range(self.n_features)]
        self.class_names = class_names
        self.mode = mode
        self.discretize_continuous = discretize_continuous
        self.random_state = random_state
        
        # Set default kernel width
        self.kernel_width = kernel_width or np.sqrt(self.n_features) * 0.75
        
        # Compute statistics from training data
        self._compute_statistics()
        
        # Initialize random number generator
        self.rng = np.random.RandomState(random_state)
        
    def _compute_statistics(self):
        """Compute statistics needed for generating perturbations."""
        # TODO: Implement computation of:
        # - self.feature_means: mean of each feature
        # - self.feature_stds: standard deviation of each feature
        # - self.feature_mins: minimum value of each feature
        # - self.feature_maxs: maximum value of each feature
        # - self.feature_ranges: range (max - min) of each feature
        
        # These statistics will be used to generate realistic perturbations
        raise NotImplementedError("Implement feature statistics computation")
        
    def _generate_perturbations(
        self,
        instance: np.ndarray,
        n_samples: int = 5000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate perturbations around the instance.
        
        Args:
            instance: single instance to explain (1D array)
            n_samples: number of perturbations to generate
            
        Returns:
            perturbations: binary matrix indicating which features are "on" (n_samples, n_features)
            perturbed_data: actual perturbed instances (n_samples, n_features)
        """
        # TODO: Implement perturbation generation
        # 1. Create binary perturbation matrix (which features to keep from original)
        # 2. Generate perturbed instances by:
        #    - Where perturbation = 1: keep original feature value
        #    - Where perturbation = 0: sample from training data distribution
        # 
        # Hints:
        # - Use self.rng.randint(0, 2, size=...) for binary matrix
        # - Sample from training data or use statistics for replacement values
        
        raise NotImplementedError("Implement perturbation generation")
        
    def _compute_kernel_weights(
        self,
        distances: np.ndarray
    ) -> np.ndarray:
        """
        Compute kernel weights based on distances.
        
        Uses exponential kernel: exp(-(distance^2) / kernel_width^2)
        
        Args:
            distances: distances from perturbations to original instance
            
        Returns:
            weights: kernel weights for each perturbation
        """
        # TODO: Implement exponential kernel weighting
        # The kernel should give higher weights to perturbations closer to the instance
        
        raise NotImplementedError("Implement kernel weighting")
        
    def explain_instance(
        self,
        instance: np.ndarray,
        predict_fn: Callable,
        labels: Optional[list] = None,
        n_features: int = 10,
        n_samples: int = 5000,
        distance_metric: str = 'euclidean'
    ) -> 'LimeExplanation':
        """
        Generate explanation for a single instance.
        
        Args:
            instance: instance to explain
            predict_fn: function that takes instances and returns predictions
            labels: labels to explain (default: all)
            n_features: maximum number of features in explanation
            n_samples: number of perturbations to generate
            distance_metric: distance metric to use
            
        Returns:
            LimeExplanation object
        """
        # Ensure instance is 1D
        instance = instance.ravel()
        
        # Generate perturbations
        perturbations, perturbed_data = self._generate_perturbations(instance, n_samples)
        
        # Get predictions for perturbed instances
        if self.mode == 'classification':
            predictions = predict_fn(perturbed_data)
            if len(predictions.shape) == 1:  # Binary classification
                predictions = np.column_stack([1 - predictions, predictions])
        else:
            predictions = predict_fn(perturbed_data).ravel()
            
        # TODO: Implement the main LIME algorithm:
        # 1. Compute distances between perturbations and original instance
        #    (use the binary perturbation matrix, not the actual data)
        # 2. Compute kernel weights based on distances
        # 3. For each label/output:
        #    a. Fit weighted linear model (Ridge regression)
        #    b. Extract feature importance (coefficients)
        # 4. Create and return LimeExplanation object
        
        # Placeholder for distances computation
        distances = None  # TODO: Compute distances using perturbations matrix
        
        # Placeholder for kernel weights
        weights = None  # TODO: Compute kernel weights
        
        # Placeholder for storing explanations
        explanations = {}
        
        # Fit weighted linear model for each label
        if labels is None:
            if self.mode == 'classification':
                labels = range(predictions.shape[1])
            else:
                labels = [0]  # Single output for regression
                
        for label in labels:
            # TODO: Implement weighted Ridge regression
            # 1. Extract target values (predictions for this label)
            # 2. Fit Ridge model with sample weights
            # 3. Store coefficients as feature importance
            
            raise NotImplementedError("Implement weighted linear model fitting")
            
        return LimeExplanation(
            instance=instance,
            explanations=explanations,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=self.mode,
            n_features=n_features
        )


class LimeExplanation:
    """Container for LIME explanation results."""
    
    def __init__(
        self,
        instance: np.ndarray,
        explanations: dict,
        feature_names: list,
        class_names: Optional[list],
        mode: str,
        n_features: int
    ):
        self.instance = instance
        self.explanations = explanations
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode
        self.n_features = n_features
        
    def as_list(self, label: Optional[int] = None) -> list:
        """
        Return explanation as a list of (feature, importance) tuples.
        
        Args:
            label: label to explain (for classification)
            
        Returns:
            List of (feature_name, importance) sorted by absolute importance
        """
        if self.mode == 'classification' and label is None:
            raise ValueError("Must specify label for classification")
        if self.mode == 'regression':
            label = 0
            
        # TODO: Implement extraction of top features
        # 1. Get feature importances for specified label
        # 2. Sort by absolute importance
        # 3. Return top n_features as list of tuples
        
        raise NotImplementedError("Implement feature list extraction")
        
    def as_pyplot_figure(self, label: Optional[int] = None):
        """
        Generate matplotlib figure showing feature importances.
        
        Args:
            label: label to explain (for classification)
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        # Get feature importance list
        features = self.as_list(label)
        
        # TODO: Implement visualization
        # 1. Create horizontal bar plot
        # 2. Color bars based on positive/negative contribution
        # 3. Add proper labels and title
        
        raise NotImplementedError("Implement visualization")


# Example usage and testing code
if __name__ == "__main__":
    # Generate synthetic dataset for testing
    np.random.seed(42)
    n_samples, n_features = 1000, 5
    X = np.random.randn(n_samples, n_features)
    
    # Create a simple linear model for ground truth
    true_weights = np.array([2.0, -1.5, 0.5, 0.0, -0.8])
    y_continuous = X @ true_weights + 0.1 * np.random.randn(n_samples)
    y_binary = (y_continuous > 0).astype(int)
    
    # Initialize LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X,
        feature_names=[f'x{i}' for i in range(n_features)],
        mode='classification',
        random_state=42
    )
    
    # Create a simple predict function
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42)
    model.fit(X, y_binary)
    
    def predict_proba(instances):
        return model.predict_proba(instances)
    
    # Explain a single instance
    instance_idx = 0
    instance = X[instance_idx]
    
    # This will fail until you implement the required methods
    try:
        explanation = explainer.explain_instance(
            instance=instance,
            predict_fn=predict_proba,
            n_features=5,
            n_samples=1000
        )
        
        # Display results
        print(f"True label: {y_binary[instance_idx]}")
        print(f"Predicted probabilities: {predict_proba(instance.reshape(1, -1))[0]}")
        print("\nFeature importances:")
        for feature, importance in explanation.as_list(label=1):
            print(f"  {feature}: {importance:.3f}")
            
    except NotImplementedError as e:
        print(f"Implementation needed: {e}")
        print("\nExpected ground truth weights:", true_weights)
        print("Your implementation should recover similar relative importances!")


# Additional utilities to implement later
class LimeImageExplainer:
    """LIME for image data - to be implemented."""
    pass


class LimeTextExplainer:
    """LIME for text data - to be implemented."""
    pass