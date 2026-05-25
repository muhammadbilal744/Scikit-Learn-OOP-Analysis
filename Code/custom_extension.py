# ============================================================
# File    : custom_extension.py
# Project : Object-Oriented Analysis of Scikit-learn Library
# Authors :
#           Muhammad Bilal     — F25BDATS1M02098
#           Maryam Tahir       — F25BDATS1M02051
#           Jaweria Zafar      — F25BDATS1M02090
#
# Subject  : Object-Oriented Programming
# Semester : BS Data Science — 2nd Semester
# Session  : 2026
# University: The Islamia University of Bahawalpur
#
# Description:
#   This file demonstrates OOP principles by extending
#   Scikit-learn's existing base classes. Three custom
#   classes are created:
#
#   1. SmartClassifier  — Inherits ClassifierMixin + BaseEstimator
#   2. SmartTransformer — Inherits TransformerMixin + BaseEstimator
#   3. SmartPipeline    — Composes both classes (Composition pattern)
#
# OOP Concepts Demonstrated:
#   - Inheritance    : Multiple parent classes inherited
#   - Encapsulation  : Internal state hidden inside classes
#   - Polymorphism   : fit(), predict(), transform() overridden
#   - Abstraction    : Complex steps hidden from user
#   - Composition    : SmartPipeline owns both classes inside
#
# How to Run:
#   pip install scikit-learn numpy
#   python custom_extension.py
#
# GitHub:
#   https://github.com/muhammadbilal744/Scikit-Learn-OOP-Analysis
# ============================================================


# ── Imports ──────────────────────────────────────────────────
import numpy as np  # For numerical operations and array handling

# Importing Scikit-learn base classes that we will inherit from
# BaseEstimator  : Provides get_params() and set_params() to all models
# ClassifierMixin: Provides score() method for classification models
# TransformerMixin: Provides fit_transform() for data transformers
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


# ============================================================
# CLASS 1 — SmartClassifier
#
# Purpose  : A custom classifier that predicts based on
#            majority class or first class strategy.
#
# Inherits : ClassifierMixin + BaseEstimator (Scikit-learn)
#
# OOP Concepts:
#   - Inheritance   : Two Scikit-learn classes inherited
#   - Encapsulation : Parameters hidden inside __init__
#   - Polymorphism  : fit() and predict() overridden
#   - Abstraction   : score() inherited — user does not see logic
# ============================================================

class SmartClassifier(ClassifierMixin, BaseEstimator):
    """
    A custom classifier that extends Scikit-learn's
    BaseEstimator and ClassifierMixin.

    This class demonstrates OOP principles by inheriting
    from two Scikit-learn base classes and adding new
    meaningful functionality.

    Parameters
    ----------
    strategy : str, default='majority'
        Strategy used for prediction.
        'majority'    — always predict the most frequent class
        'first_class' — always predict the first class found

    threshold : float, default=0.5
        Decision threshold (stored for future use).

    Examples
    --------
    >>> clf = SmartClassifier(strategy='majority')
    >>> clf.fit(X, y)
    >>> clf.predict(X)
    array([0, 0, 0, 0, 0])
    """

    def __init__(self, strategy='majority', threshold=0.5):
        """
        Initialize SmartClassifier with given parameters.

        This is the constructor — called when object is created.
        Parameters are stored inside the object (Encapsulation).

        Parameters
        ----------
        strategy  : str   — prediction strategy
        threshold : float — decision threshold
        """
        # Encapsulation — storing parameters inside the object
        # These are user-set hyperparameters (no trailing underscore)
        self.strategy  = strategy
        self.threshold = threshold

    def fit(self, X, y):
        """
        Train the model by learning from data.

        This method OVERRIDES the inherited fit() concept
        from BaseEstimator. This is Polymorphism — same
        method name, our own custom implementation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix
        y : array-like of shape (n_samples,)
            Target labels (0 or 1)

        Returns
        -------
        self : SmartClassifier
            Returns self for method chaining (sklearn convention)
        """
        # Learn unique classes from training data
        # Trailing underscore (_) means this is a LEARNED parameter
        self.classes_       = np.unique(y)

        # Count how many classes we have
        self.n_classes_     = len(self.classes_)

        # Store number of input features
        self.n_features_in_ = X.shape[1]

        # Count occurrences of each class
        counts              = np.bincount(y.astype(int))

        # Store class counts for inspection
        self.class_counts_  = counts

        # Find which class appears most often (majority class)
        self.majority_class_= int(np.argmax(counts))

        # Mark model as fitted — used to check in predict()
        self.is_fitted_     = True

        # Always return self — required by Scikit-learn convention
        return self

    def predict(self, X):
        """
        Predict class labels for input samples.

        This method OVERRIDES the predict() concept from
        BaseEstimator. This is Polymorphism — same name,
        our own implementation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix to predict on

        Returns
        -------
        predictions : numpy array of shape (n_samples,)
            Predicted class label for each sample
        """
        # Check that fit() was called before predict()
        # This is Encapsulation — protecting internal state
        if not hasattr(self, 'is_fitted_'):
            raise Exception(
                "Model not fitted! Call fit(X, y) before predict(X)."
            )

        # Apply prediction strategy chosen by user
        if self.strategy == 'majority':
            # Return majority class for every sample
            return np.full(len(X), self.majority_class_)

        elif self.strategy == 'first_class':
            # Return first class found in training data
            return np.full(len(X), self.classes_[0])

        else:
            # Default fallback — use majority class
            return np.full(len(X), self.majority_class_)

    # ── New Methods (not in any parent class) ────────────────

    def describe(self):
        """
        NEW METHOD — does not exist in any parent class.

        Print a human-readable description of this model.
        Shows current state, parameters, and learned values.
        """
        print("=" * 45)
        print("      SmartClassifier Description")
        print("=" * 45)
        print(f"  Strategy    : {self.strategy}")
        print(f"  Threshold   : {self.threshold}")

        # Show learned parameters only if model is fitted
        if hasattr(self, 'is_fitted_'):
            print(f"  Status      : Fitted")
            print(f"  Classes     : {self.classes_}")
            print(f"  Class Counts: {self.class_counts_}")
            print(f"  Majority    : Class {self.majority_class_}")
            print(f"  Features    : {self.n_features_in_}")
        else:
            print(f"  Status      : Not fitted yet")
        print("=" * 45)

    def get_model_info(self):
        """
        NEW METHOD — does not exist in any parent class.

        Return a dictionary containing model information.
        Useful for logging and inspection.

        Returns
        -------
        info : dict
            Dictionary with model name, parents, and parameters
        """
        return {
            'model_name'    : 'SmartClassifier',
            # Shows which Scikit-learn classes we inherited from
            'inherits_from' : ['ClassifierMixin', 'BaseEstimator'],
            'strategy'      : self.strategy,
            'threshold'     : self.threshold,
            'is_fitted'     : hasattr(self, 'is_fitted_'),
            # getattr used safely — returns None if not fitted yet
            'n_classes'     : getattr(self, 'n_classes_', None),
            'majority_class': getattr(self, 'majority_class_', None),
        }

    def compare_strategies(self, X, y):
        """
        NEW METHOD — does not exist in any parent class.

        Compare accuracy of both prediction strategies
        and show which one performs better on given data.

        Parameters
        ----------
        X : array-like — feature matrix
        y : array-like — true labels
        """
        # Ensure model is fitted before comparing
        if not hasattr(self, 'is_fitted_'):
            raise Exception("Call fit() before compare_strategies()!")

        # Calculate accuracy for majority strategy
        majority_preds = np.full(len(X), self.majority_class_)
        majority_score = np.mean(majority_preds == y)

        # Calculate accuracy for first_class strategy
        first_preds    = np.full(len(X), self.classes_[0])
        first_score    = np.mean(first_preds == y)

        # Display comparison results
        print("=" * 45)
        print("        Strategy Comparison")
        print("=" * 45)
        print(f"  Majority Class : {majority_score:.2f} accuracy")
        print(f"  First Class    : {first_score:.2f} accuracy")
        winner = "Majority" if majority_score >= first_score else "First Class"
        print(f"  Winner         : {winner} strategy")
        print("=" * 45)

    def __str__(self):
        """
        Dunder (magic) method — called when print(object) is used.
        Returns a clean string representation of this object.
        """
        return (
            f"SmartClassifier("
            f"strategy='{self.strategy}', "
            f"threshold={self.threshold})"
        )

    def __repr__(self):
        """
        Dunder method — called in interactive Python sessions.
        Returns official string representation.
        """
        return self.__str__()


# ============================================================
# CLASS 2 — SmartTransformer
#
# Purpose  : A custom data transformer that scales features
#            using standard scaling or min-max scaling.
#
# Inherits : TransformerMixin + BaseEstimator (Scikit-learn)
#
# OOP Concepts:
#   - Inheritance   : Two Scikit-learn classes inherited
#   - Encapsulation : mean_, std_, min_, max_ stored inside
#   - Polymorphism  : fit() and transform() overridden
#   - Abstraction   : fit_transform() inherited — hides steps
# ============================================================

class SmartTransformer(TransformerMixin, BaseEstimator):
    """
    A custom data transformer that extends Scikit-learn's
    TransformerMixin and BaseEstimator.

    Scales input features using standard or minmax method.
    fit_transform() is automatically inherited from
    TransformerMixin — this is Abstraction in action.

    Parameters
    ----------
    method : str, default='standard'
        Scaling method to apply.
        'standard' — zero mean, unit variance (Z-score)
        'minmax'   — scale to range [0, 1]

    Examples
    --------
    >>> t = SmartTransformer(method='standard')
    >>> X_scaled = t.fit_transform(X)
    """

    def __init__(self, method='standard'):
        """
        Initialize SmartTransformer with scaling method.

        Parameters
        ----------
        method : str — 'standard' or 'minmax'
        """
        # Encapsulation — storing parameter inside object
        self.method = method

    def fit(self, X, y=None):
        """
        Learn statistics from training data.

        This method OVERRIDES fit() from TransformerMixin.
        This is Polymorphism — same name, our implementation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to learn statistics from
        y : ignored — present for API compatibility only

        Returns
        -------
        self : SmartTransformer
        """
        # Learn and store statistics from training data
        # These are LEARNED parameters — trailing underscore (_)
        # Encapsulation — stored inside the object

        self.mean_      = np.mean(X, axis=0)  # Mean of each feature
        self.std_       = np.std(X, axis=0)   # Std deviation of each feature
        self.min_       = np.min(X, axis=0)   # Minimum of each feature
        self.max_       = np.max(X, axis=0)   # Maximum of each feature

        # Mark as fitted
        self.is_fitted_ = True

        # Return self — required by Scikit-learn convention
        return self

    def transform(self, X):
        """
        Apply learned scaling to new data.

        This method OVERRIDES transform() from TransformerMixin.
        This is Polymorphism — same name, our implementation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform using learned statistics

        Returns
        -------
        X_transformed : numpy array of same shape as X
        """
        # Check fit() was called first
        if not hasattr(self, 'is_fitted_'):
            raise Exception(
                "Transformer not fitted! Call fit(X) before transform(X)."
            )

        if self.method == 'standard':
            # Standard (Z-score) scaling:
            # Formula: (X - mean) / std
            # Result: mean=0, std=1 for each feature
            return (X - self.mean_) / (self.std_ + 1e-8)
            # 1e-8 added to prevent division by zero

        elif self.method == 'minmax':
            # MinMax scaling:
            # Formula: (X - min) / (max - min)
            # Result: all values between 0 and 1
            return (X - self.min_) / (self.max_ - self.min_ + 1e-8)

        else:
            # If unknown method — return data unchanged
            return X

    # ── New Methods (not in any parent class) ────────────────

    def describe(self):
        """
        NEW METHOD — does not exist in any parent class.
        Print transformer state and learned statistics.
        """
        print("=" * 45)
        print("     SmartTransformer Description")
        print("=" * 45)
        print(f"  Method  : {self.method}")
        if hasattr(self, 'is_fitted_'):
            print(f"  Status  : Fitted")
            print(f"  Mean    : {self.mean_.round(2)}")
            print(f"  Std     : {self.std_.round(2)}")
            print(f"  Min     : {self.min_.round(2)}")
            print(f"  Max     : {self.max_.round(2)}")
        else:
            print(f"  Status  : Not fitted yet")
        print("=" * 45)

    def get_transformer_info(self):
        """
        NEW METHOD — does not exist in any parent class.
        Return transformer information as dictionary.
        """
        return {
            'model_name'    : 'SmartTransformer',
            'inherits_from' : ['TransformerMixin', 'BaseEstimator'],
            'method'        : self.method,
            'is_fitted'     : hasattr(self, 'is_fitted_'),
        }

    def __str__(self):
        """Dunder method — string representation."""
        return f"SmartTransformer(method='{self.method}')"

    def __repr__(self):
        """Dunder method — official representation."""
        return self.__str__()


# ============================================================
# CLASS 3 — SmartPipeline
#
# Purpose  : Combines SmartTransformer and SmartClassifier
#            into one complete ML workflow.
#
# Inherits : None — uses Composition pattern instead
#
# OOP Concepts:
#   - Composition  : Contains SmartTransformer + SmartClassifier
#   - Abstraction  : User calls fit/predict — steps are hidden
#   - Encapsulation: Both objects stored and protected inside
# ============================================================

class SmartPipeline:
    """
    A custom pipeline that combines SmartTransformer and
    SmartClassifier into one complete workflow.

    This class demonstrates the COMPOSITION design pattern —
    it CONTAINS objects of other classes rather than
    inheriting from them.

    Abstraction in action:
    User simply calls fit() and predict() — the pipeline
    internally handles all the steps automatically.

    Parameters
    ----------
    transformer : SmartTransformer
        Transformer object to preprocess the data

    classifier : SmartClassifier
        Classifier object to make predictions

    Examples
    --------
    >>> pipeline = SmartPipeline(
    ...     transformer=SmartTransformer(),
    ...     classifier=SmartClassifier()
    ... )
    >>> pipeline.fit(X, y)
    >>> pipeline.predict(X)
    """

    def __init__(self, transformer, classifier):
        """
        Initialize pipeline with transformer and classifier.

        Composition — both objects are stored INSIDE this class.
        This is different from Inheritance — we own these objects
        rather than being a child of their classes.

        Parameters
        ----------
        transformer : SmartTransformer instance
        classifier  : SmartClassifier instance
        """
        # Composition — storing objects inside this class
        # These objects will be used in fit() and predict()
        self.transformer = transformer
        self.classifier  = classifier

        # Pipeline starts as not fitted
        self.is_fitted_  = False

    def fit(self, X, y):
        """
        Fit the complete pipeline on training data.

        This is ABSTRACTION — the user calls one fit() method
        but internally two steps happen automatically:
          Step 1: Transformer learns statistics and scales data
          Step 2: Classifier learns from scaled data

        Parameters
        ----------
        X : array-like — raw feature matrix
        y : array-like — target labels

        Returns
        -------
        self : SmartPipeline
        """
        # Step 1 — Fit transformer and transform data
        # fit_transform() is inherited from TransformerMixin
        # This is Abstraction — internally calls fit() + transform()
        X_transformed = self.transformer.fit_transform(X, y)

        # Step 2 — Train classifier on the transformed data
        self.classifier.fit(X_transformed, y)

        # Mark pipeline as fitted
        self.is_fitted_ = True

        return self  # Return self for method chaining

    def predict(self, X):
        """
        Make predictions on new data.

        Abstraction — user calls predict(), pipeline handles:
          Step 1: Transform new data using learned statistics
          Step 2: Predict using trained classifier

        Parameters
        ----------
        X : array-like — raw feature matrix

        Returns
        -------
        predictions : numpy array of class labels
        """
        # Ensure pipeline was fitted first
        if not self.is_fitted_:
            raise Exception(
                "Pipeline not fitted! Call fit(X, y) before predict(X)."
            )

        # Step 1 — Transform new data (use learned statistics)
        X_transformed = self.transformer.transform(X)

        # Step 2 — Predict using trained classifier
        return self.classifier.predict(X_transformed)

    def score(self, X, y):
        """
        NEW METHOD — Calculate overall pipeline accuracy.

        Predicts on X and compares with true labels y.

        Parameters
        ----------
        X : array-like — feature matrix
        y : array-like — true labels

        Returns
        -------
        accuracy : float between 0.0 and 1.0
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def describe(self):
        """
        NEW METHOD — Print complete pipeline description.
        Shows both transformer and classifier details.
        """
        print("=" * 45)
        print("       SmartPipeline Description")
        print("=" * 45)
        print(f"  Transformer : {self.transformer}")
        print(f"  Classifier  : {self.classifier}")
        print(f"  Status      : {'Fitted' if self.is_fitted_ else 'Not fitted yet'}")
        print("=" * 45)

        # Show details of each component
        print("  Step 1 — SmartTransformer:")
        self.transformer.describe()

        print("  Step 2 — SmartClassifier:")
        self.classifier.describe()

    def get_pipeline_info(self):
        """
        NEW METHOD — Return pipeline information as dictionary.
        """
        return {
            'pipeline_name' : 'SmartPipeline',
            'transformer'   : str(self.transformer),
            'classifier'    : str(self.classifier),
            'is_fitted'     : self.is_fitted_,
            'oop_concept'   : 'Composition',
        }

    def __str__(self):
        """Dunder method — string representation."""
        return (
            f"SmartPipeline("
            f"transformer={self.transformer}, "
            f"classifier={self.classifier})"
        )

    def __repr__(self):
        """Dunder method — official representation."""
        return self.__str__()


# ============================================================
# DEMO — Main block
#
# This section runs when file is executed directly:
#   python custom_extension.py
#
# It demonstrates all 3 classes working together and
# shows which OOP concepts are being used at each step.
# ============================================================

if __name__ == '__main__':

    # ── Sample Training Data ──────────────────────────────────
    # X = feature matrix (5 samples, 2 features each)
    # y = target labels  (3 zeros, 2 ones)
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0]
    ])
    y = np.array([0, 1, 0, 0, 1])

    # Print demo header
    print("\n" + "=" * 45)
    print("   3 Classes Demo — Scikit-learn OOP")
    print("=" * 45 + "\n")

    # ════════════════════════════════════════
    # CLASS 1 DEMO — SmartClassifier
    # ════════════════════════════════════════
    print("CLASS 1 — SmartClassifier")
    print("  Inherits: ClassifierMixin + BaseEstimator")
    print("-" * 45)

    # Create object — calls __init__()
    clf = SmartClassifier(strategy='majority', threshold=0.5)
    print(f"  Object      : {clf}")  # Calls __str__()

    # fit() — our overridden method (Polymorphism)
    clf.fit(X, y)
    print(f"  fit()       : Training complete (Polymorphism)")

    # predict() — our overridden method (Polymorphism)
    predictions = clf.predict(X)
    print(f"  predict()   : {predictions} (Polymorphism)")

    # score() — FREE from ClassifierMixin (Inheritance + Abstraction)
    accuracy = clf.score(X, y)
    print(f"  score()     : {accuracy:.2f}  <- Inherited from ClassifierMixin")

    # get_params() — FREE from BaseEstimator (Inheritance)
    params = clf.get_params()
    print(f"  get_params(): {params}  <- Inherited from BaseEstimator")

    # New custom methods
    clf.describe()           # New method — not in parent
    clf.compare_strategies(X, y)  # New method — not in parent
    print()

    # ════════════════════════════════════════
    # CLASS 2 DEMO — SmartTransformer
    # ════════════════════════════════════════
    print("CLASS 2 — SmartTransformer")
    print("  Inherits: TransformerMixin + BaseEstimator")
    print("-" * 45)

    # Create transformer object
    transformer = SmartTransformer(method='standard')
    print(f"  Object         : {transformer}")

    # fit_transform() — FREE from TransformerMixin (Abstraction)
    # Internally calls fit() then transform() automatically
    X_transformed = transformer.fit_transform(X)
    print(f"  fit_transform(): Inherited from TransformerMixin (Abstraction)")
    print(f"  Original  X[0] : {X[0]}")
    print(f"  Transformed[0] : {X_transformed[0].round(3)}")

    # get_params() — FREE from BaseEstimator (Inheritance)
    params = transformer.get_params()
    print(f"  get_params()   : {params}  <- Inherited from BaseEstimator")

    # New custom methods
    transformer.describe()  # New method — not in parent
    print()

    # ════════════════════════════════════════
    # CLASS 3 DEMO — SmartPipeline
    # ════════════════════════════════════════
    print("CLASS 3 — SmartPipeline")
    print("  Pattern: Composition (owns Transformer + Classifier)")
    print("-" * 45)

    # Create pipeline — Composition pattern
    # Pipeline OWNS both objects inside it
    pipeline = SmartPipeline(
        transformer = SmartTransformer(method='standard'),
        classifier  = SmartClassifier(strategy='majority')
    )
    print(f"  Object    : SmartPipeline created (Composition)")

    # fit() — internally transforms then trains (Abstraction)
    pipeline.fit(X, y)
    print(f"  fit()     : Step1=Transform, Step2=Train (Abstraction)")

    # predict() — internally transforms then predicts (Abstraction)
    preds = pipeline.predict(X)
    print(f"  predict() : {preds} (Abstraction)")

    # score() — our new method
    score = pipeline.score(X, y)
    print(f"  score()   : {score:.2f}")

    # New custom method
    pipeline.describe()

    # ── Final Summary ─────────────────────────────────────────
    print("\n" + "=" * 45)
    print("  All 3 Classes Demo Complete!")
    print("=" * 45)
    print("  OOP Concepts Demonstrated:")
    print("    Inheritance    : SmartClassifier, SmartTransformer")
    print("    Encapsulation  : All private attributes with _")
    print("    Polymorphism   : fit(), predict(), transform()")
    print("    Abstraction    : fit_transform(), score()")
    print("    Composition    : SmartPipeline owns both classes")
    print("=" * 45)