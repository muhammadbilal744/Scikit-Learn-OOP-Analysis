# ============================================================
# custom_extension.py
# Scikit-learn OOP Analysis — Custom Extension
#
#
#
# Library  : Scikit-learn
# Subject  : Object-Oriented Programming
# Semester : BS Data Science — 2nd Semester
#
# Classes Created:
#   1. SmartClassifier  — ClassifierMixin + BaseEstimator
#   2. SmartTransformer — TransformerMixin + BaseEstimator
#   3. SmartPipeline    — Composition of above two classes
#
# OOP Concepts Demonstrated:
#   - Inheritance    : Multiple parent classes inherited
#   - Encapsulation  : Internal state hidden inside classes
#   - Polymorphism   : fit(), predict(), transform() overridden
#   - Abstraction    : User sees simple interface
#   - Composition    : SmartPipeline uses both classes inside
# ============================================================

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


# ============================================================
# CLASS 1 — SmartClassifier
# Inherits: ClassifierMixin + BaseEstimator
# ============================================================

class SmartClassifier(ClassifierMixin, BaseEstimator):
    """
    A custom classifier that extends Scikit-learn's
    BaseEstimator and ClassifierMixin.

    OOP Concepts:
    - Inheritance   : ClassifierMixin + BaseEstimator inherited
    - Encapsulation : Parameters stored inside object
    - Polymorphism  : fit() and predict() overridden
    - Abstraction   : score() hidden inside ClassifierMixin

    Parameters
    ----------
    strategy : str, default='majority'
        Options: 'majority', 'first_class'
    threshold : float, default=0.5
        Decision threshold.
    """

    def __init__(self, strategy='majority', threshold=0.5):
        # Encapsulation — parameters stored inside object
        self.strategy  = strategy
        self.threshold = threshold

    def fit(self, X, y):
        """Train the model — override from BaseEstimator."""
        self.classes_        = np.unique(y)
        self.n_classes_      = len(self.classes_)
        self.n_features_in_  = X.shape[1]
        counts               = np.bincount(y.astype(int))
        self.class_counts_   = counts
        self.majority_class_ = int(np.argmax(counts))
        self.is_fitted_      = True
        return self

    def predict(self, X):
        """Predict class labels — override from BaseEstimator."""
        if not hasattr(self, 'is_fitted_'):
            raise Exception("Call fit() before predict()!")
        if self.strategy == 'majority':
            return np.full(len(X), self.majority_class_)
        elif self.strategy == 'first_class':
            return np.full(len(X), self.classes_[0])
        else:
            return np.full(len(X), self.majority_class_)

    def describe(self):
        """NEW method — print model description."""
        print("=" * 45)
        print("      SmartClassifier Description")
        print("=" * 45)
        print(f"  Strategy    : {self.strategy}")
        print(f"  Threshold   : {self.threshold}")
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
        """NEW method — return model info as dictionary."""
        return {
            'model_name'    : 'SmartClassifier',
            'inherits_from' : ['ClassifierMixin', 'BaseEstimator'],
            'strategy'      : self.strategy,
            'threshold'     : self.threshold,
            'is_fitted'     : hasattr(self, 'is_fitted_'),
            'n_classes'     : getattr(self, 'n_classes_', None),
            'majority_class': getattr(self, 'majority_class_', None),
        }

    def compare_strategies(self, X, y):
        """NEW method — compare both strategies."""
        if not hasattr(self, 'is_fitted_'):
            raise Exception("Call fit() before compare_strategies()!")
        majority_score = np.mean(np.full(len(X), self.majority_class_) == y)
        first_score    = np.mean(np.full(len(X), self.classes_[0]) == y)
        print("=" * 45)
        print("        Strategy Comparison")
        print("=" * 45)
        print(f"  Majority Class : {majority_score:.2f} accuracy")
        print(f"  First Class    : {first_score:.2f} accuracy")
        winner = "Majority" if majority_score >= first_score else "First Class"
        print(f"  Winner         : {winner} strategy")
        print("=" * 45)

    def __str__(self):
        return f"SmartClassifier(strategy='{self.strategy}', threshold={self.threshold})"

    def __repr__(self):
        return self.__str__()


# ============================================================
# CLASS 2 — SmartTransformer
# Inherits: TransformerMixin + BaseEstimator
# ============================================================

class SmartTransformer(TransformerMixin, BaseEstimator):
    """
    A custom transformer that extends Scikit-learn's
    TransformerMixin and BaseEstimator.

    OOP Concepts:
    - Inheritance   : TransformerMixin + BaseEstimator inherited
    - Encapsulation : mean_ and std_ stored inside
    - Polymorphism  : fit() and transform() overridden
    - Abstraction   : fit_transform() hidden in TransformerMixin

    Parameters
    ----------
    method : str, default='standard'
        Options: 'standard', 'minmax'
    """

    def __init__(self, method='standard'):
        self.method = method

    def fit(self, X, y=None):
        """Learn statistics — override from TransformerMixin."""
        self.mean_      = np.mean(X, axis=0)
        self.std_       = np.std(X, axis=0)
        self.min_       = np.min(X, axis=0)
        self.max_       = np.max(X, axis=0)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform data — override from TransformerMixin."""
        if not hasattr(self, 'is_fitted_'):
            raise Exception("Call fit() before transform()!")
        if self.method == 'standard':
            return (X - self.mean_) / (self.std_ + 1e-8)
        elif self.method == 'minmax':
            return (X - self.min_) / (self.max_ - self.min_ + 1e-8)
        else:
            return X

    def describe(self):
        """NEW method — print transformer description."""
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
        """NEW method — return transformer info as dictionary."""
        return {
            'model_name'    : 'SmartTransformer',
            'inherits_from' : ['TransformerMixin', 'BaseEstimator'],
            'method'        : self.method,
            'is_fitted'     : hasattr(self, 'is_fitted_'),
        }

    def __str__(self):
        return f"SmartTransformer(method='{self.method}')"

    def __repr__(self):
        return self.__str__()


# ============================================================
# CLASS 3 — SmartPipeline
# Composition of SmartTransformer + SmartClassifier
# ============================================================

class SmartPipeline:
    """
    A custom pipeline that composes SmartTransformer
    and SmartClassifier into one complete workflow.

    OOP Concepts:
    - Composition  : SmartTransformer + SmartClassifier inside
    - Abstraction  : User calls fit/predict — details hidden
    - Encapsulation: Both objects stored inside pipeline

    Parameters
    ----------
    transformer : SmartTransformer
    classifier  : SmartClassifier
    """

    def __init__(self, transformer, classifier):
        # Composition — objects stored inside
        self.transformer = transformer
        self.classifier  = classifier
        self.is_fitted_  = False

    def fit(self, X, y):
        """Fit transformer then classifier."""
        X_transformed   = self.transformer.fit_transform(X, y)
        self.classifier.fit(X_transformed, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Transform then predict."""
        if not self.is_fitted_:
            raise Exception("Call fit() before predict()!")
        X_transformed = self.transformer.transform(X)
        return self.classifier.predict(X_transformed)

    def score(self, X, y):
        """Calculate pipeline accuracy."""
        return np.mean(self.predict(X) == y)

    def describe(self):
        """NEW method — print full pipeline description."""
        print("=" * 45)
        print("       SmartPipeline Description")
        print("=" * 45)
        print(f"  Transformer : {self.transformer}")
        print(f"  Classifier  : {self.classifier}")
        print(f"  Status      : {'Fitted' if self.is_fitted_ else 'Not fitted yet'}")
        print("=" * 45)
        print("  Step 1 — SmartTransformer:")
        self.transformer.describe()
        print("  Step 2 — SmartClassifier:")
        self.classifier.describe()

    def get_pipeline_info(self):
        """NEW method — return pipeline info as dictionary."""
        return {
            'pipeline_name' : 'SmartPipeline',
            'transformer'   : str(self.transformer),
            'classifier'    : str(self.classifier),
            'is_fitted'     : self.is_fitted_,
            'oop_concept'   : 'Composition',
        }

    def __str__(self):
        return f"SmartPipeline(transformer={self.transformer}, classifier={self.classifier})"

    def __repr__(self):
        return self.__str__()


# ============================================================
# DEMO — Run this file to see all 3 classes in action
# ============================================================
if __name__ == '__main__':

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0]
    ])
    y = np.array([0, 1, 0, 0, 1])

    print("\n" + "=" * 45)
    print("   3 Classes Demo — Scikit-learn OOP")
    print("=" * 45 + "\n")

    # ── CLASS 1: SmartClassifier ──
    print("CLASS 1 — SmartClassifier")
    print("-" * 45)
    clf = SmartClassifier(strategy='majority', threshold=0.5)
    print(f"  Object      : {clf}")
    clf.fit(X, y)
    print(f"  predict()   : {clf.predict(X)}")
    print(f"  score()     : {clf.score(X, y):.2f}  <- from ClassifierMixin")
    print(f"  get_params(): {clf.get_params()}  <- from BaseEstimator")
    clf.describe()
    clf.compare_strategies(X, y)
    print()

    # ── CLASS 2: SmartTransformer ──
    print("CLASS 2 — SmartTransformer")
    print("-" * 45)
    transformer = SmartTransformer(method='standard')
    print(f"  Object         : {transformer}")
    X_transformed = transformer.fit_transform(X)
    print(f"  fit_transform(): from TransformerMixin")
    print(f"  Original  X[0] : {X[0]}")
    print(f"  Transformed[0] : {X_transformed[0].round(3)}")
    print(f"  get_params()   : {transformer.get_params()}  <- from BaseEstimator")
    transformer.describe()
    print()

    # ── CLASS 3: SmartPipeline ──
    print("CLASS 3 — SmartPipeline")
    print("-" * 45)
    pipeline = SmartPipeline(
        transformer = SmartTransformer(method='standard'),
        classifier  = SmartClassifier(strategy='majority')
    )
    print(f"  Object    : {pipeline}")
    pipeline.fit(X, y)
    print(f"  predict() : {pipeline.predict(X)}")
    print(f"  score()   : {pipeline.score(X, y):.2f}")
    pipeline.describe()
    print()

    print("=" * 45)
    print("  All 3 Classes Demo Complete!")
    print("  OOP Concepts Demonstrated:")
    print("    - Inheritance")
    print("    - Encapsulation")
    print("    - Polymorphism")
    print("    - Abstraction")
    print("    - Composition")
    print("=" * 45)