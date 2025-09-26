"""
NASA Kepler Exoplanet Classification - Model Training Framework

This module provides a comprehensive framework for training and evaluating
multiple machine learning models on the processed NASA Kepler dataset.

Author: NASA Space Apps Challenge Team
Date: September 26, 2025
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Core ML libraries
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate, 
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Advanced ML libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Visualization and analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class ModelResults:
    """Container for model evaluation results"""
    model_name: str
    cv_scores: Dict[str, np.ndarray]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    best_params: Optional[Dict[str, Any]] = None
    feature_importance: Optional[np.ndarray] = None
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None


class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, name: str, random_state: int = 42):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        
    @abstractmethod
    def create_model(self) -> Any:
        """Create the underlying model instance"""
        pass
    
    @abstractmethod
    def get_hyperparameter_space(self) -> Dict[str, List]:
        """Return hyperparameter space for tuning"""
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """Fit the model to training data"""
        if self.model is None:
            self.model = self.create_model()
        
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} must be fitted before prediction")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, use decision_function if available
            if hasattr(self.model, 'decision_function'):
                decision_scores = self.model.decision_function(X)
                # Convert to probabilities using sigmoid
                proba_pos = 1 / (1 + np.exp(-decision_scores))
                return np.column_stack([1 - proba_pos, proba_pos])
            else:
                raise NotImplementedError(f"Model {self.name} doesn't support probability prediction")
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available"""
        if not self.is_fitted:
            return None
            
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        else:
            return None


class LogisticRegressionModel(BaseModel):
    """Logistic Regression with regularization"""
    
    def __init__(self, random_state: int = 42):
        super().__init__("Logistic Regression", random_state)
    
    def create_model(self) -> LogisticRegression:
        return LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
    
    def get_hyperparameter_space(self) -> Dict[str, List]:
        return {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga']
        }


class RandomForestModel(BaseModel):
    """Random Forest Classifier"""
    
    def __init__(self, random_state: int = 42):
        super().__init__("Random Forest", random_state)
    
    def create_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
    
    def get_hyperparameter_space(self) -> Dict[str, List]:
        return {
            'n_estimators': [100, 300, 500],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }


class XGBoostModel(BaseModel):
    """XGBoost Classifier"""
    
    def __init__(self, random_state: int = 42):
        super().__init__("XGBoost", random_state)
    
    def create_model(self) -> Any:
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        return xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
    
    def get_hyperparameter_space(self) -> Dict[str, List]:
        return {
            'n_estimators': [100, 300, 500, 1000],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }


class LightGBMModel(BaseModel):
    """LightGBM Classifier"""
    
    def __init__(self, random_state: int = 42):
        super().__init__("LightGBM", random_state)
    
    def create_model(self) -> Any:
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
        
        return lgb.LGBMClassifier(
            random_state=self.random_state,
            verbose=-1,
            class_weight='balanced'
        )
    
    def get_hyperparameter_space(self) -> Dict[str, List]:
        return {
            'n_estimators': [100, 300, 500],
            'num_leaves': [31, 63, 127],
            'learning_rate': [0.05, 0.1, 0.2],
            'feature_fraction': [0.8, 0.9, 1.0],
            'bagging_fraction': [0.8, 0.9, 1.0]
        }


class NeuralNetworkModel(BaseModel):
    """Multi-layer Perceptron Neural Network"""
    
    def __init__(self, random_state: int = 42):
        super().__init__("Neural Network", random_state)
    
    def create_model(self) -> MLPClassifier:
        return MLPClassifier(
            random_state=self.random_state,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1
        )
    
    def get_hyperparameter_space(self) -> Dict[str, List]:
        return {
            'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'learning_rate_init': [0.001, 0.01]
        }


class ModelTrainer:
    """Main class for training and evaluating multiple models"""
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.results = {}
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, train_path: str, test_path: str) -> 'ModelTrainer':
        """Load training and test datasets"""
        print(f"📂 Loading data...")
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Separate features and targets
        self.X_train = train_df.drop('target', axis=1)
        self.y_train = train_df['target']
        self.X_test = test_df.drop('target', axis=1)
        self.y_test = test_df['target']
        
        print(f"  ✅ Training set: {self.X_train.shape[0]:,} samples × {self.X_train.shape[1]} features")
        print(f"  ✅ Test set: {self.X_test.shape[0]:,} samples × {self.X_test.shape[1]} features")
        print(f"  📊 Class distribution (train): {np.bincount(self.y_train)}")
        print(f"  📊 Class distribution (test): {np.bincount(self.y_test)}")
        
        return self
    
    def evaluate_model(self, model: BaseModel, cv_folds: int = 5, 
                      hyperparameter_tuning: bool = False) -> ModelResults:
        """Evaluate a single model with cross-validation"""
        print(f"\n🔄 Evaluating {model.name}...")
        
        import time
        start_time = time.time()
        
        # Hyperparameter tuning if requested
        best_params = None
        if hyperparameter_tuning:
            print(f"  🔧 Performing hyperparameter tuning...")
            param_space = model.get_hyperparameter_space()
            
            # Use RandomizedSearchCV for efficiency
            search = RandomizedSearchCV(
                model.create_model(),
                param_space,
                cv=3,  # Reduced CV for speed
                n_iter=20,  # Limited iterations for speed
                scoring='accuracy',
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            
            search.fit(self.X_train, self.y_train)
            best_params = search.best_params_
            model.model = search.best_estimator_
            model.is_fitted = True
            print(f"    ✅ Best parameters: {best_params}")
        else:
            # Use default parameters
            model.fit(self.X_train, self.y_train)
        
        # Cross-validation evaluation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        scoring_metrics = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        cv_results = cross_validate(
            model.model, self.X_train, self.y_train,
            cv=cv, scoring=scoring_metrics, n_jobs=self.n_jobs
        )
        
        # Calculate mean and std for each metric
        mean_scores = {metric: np.mean(cv_results[f'test_{metric}']) for metric in scoring_metrics.keys()}
        std_scores = {metric: np.std(cv_results[f'test_{metric}']) for metric in scoring_metrics.keys()}
        
        training_time = time.time() - start_time
        
        # Test prediction time
        start_pred = time.time()
        _ = model.predict(self.X_test[:100])  # Sample for speed
        prediction_time = (time.time() - start_pred) * len(self.X_test) / 100
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        # Print results
        print(f"  📊 Cross-validation results ({cv_folds}-fold):")
        for metric, mean_score in mean_scores.items():
            std_score = std_scores[metric]
            print(f"    {metric:<10}: {mean_score:.4f} ± {std_score:.4f}")
        print(f"  ⏱️  Training time: {training_time:.2f}s")
        print(f"  ⚡ Prediction time: {prediction_time:.4f}s")
        
        return ModelResults(
            model_name=model.name,
            cv_scores={f'test_{metric}': cv_results[f'test_{metric}'] for metric in scoring_metrics.keys()},
            mean_scores=mean_scores,
            std_scores=std_scores,
            best_params=best_params,
            feature_importance=feature_importance,
            training_time=training_time,
            prediction_time=prediction_time
        )
    
    def train_all_models(self, models: List[BaseModel], cv_folds: int = 5,
                        hyperparameter_tuning: bool = False) -> Dict[str, ModelResults]:
        """Train and evaluate all provided models"""
        print("🚀 NASA Kepler Exoplanet Classification - Model Training")
        print("=" * 60)
        
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        results = {}
        
        for model in models:
            try:
                result = self.evaluate_model(model, cv_folds, hyperparameter_tuning)
                results[model.name] = result
                self.results[model.name] = result
            except Exception as e:
                print(f"❌ Error training {model.name}: {e}")
                continue
        
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models"""
        if not self.results:
            print("No models have been trained yet.")
            return pd.DataFrame()
        
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{result.mean_scores['accuracy']:.4f} ± {result.std_scores['accuracy']:.4f}",
                'Precision': f"{result.mean_scores['precision']:.4f} ± {result.std_scores['precision']:.4f}",
                'Recall': f"{result.mean_scores['recall']:.4f} ± {result.std_scores['recall']:.4f}",
                'F1-Score': f"{result.mean_scores['f1']:.4f} ± {result.std_scores['f1']:.4f}",
                'ROC-AUC': f"{result.mean_scores['roc_auc']:.4f} ± {result.std_scores['roc_auc']:.4f}",
                'Training Time': f"{result.training_time:.2f}s",
                'Prediction Time': f"{result.prediction_time:.4f}s"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.sort_values('Accuracy', ascending=False)
    
    def get_best_model(self, metric: str = 'accuracy') -> Tuple[str, ModelResults]:
        """Get the best performing model based on specified metric"""
        if not self.results:
            raise ValueError("No models have been trained yet.")
        
        best_name = max(self.results.keys(), 
                       key=lambda name: self.results[name].mean_scores[metric])
        return best_name, self.results[best_name]
    
    def save_results(self, output_dir: str) -> None:
        """Save training results and models"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save comparison table
        comparison_df = self.compare_models()
        comparison_df.to_csv(output_path / "model_comparison.csv", index=False)
        
        # Save detailed results
        results_summary = {}
        for name, result in self.results.items():
            results_summary[name] = {
                'mean_scores': result.mean_scores,
                'std_scores': result.std_scores,
                'best_params': result.best_params,
                'training_time': result.training_time,
                'prediction_time': result.prediction_time
            }
        
        with open(output_path / "training_results.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"💾 Results saved to: {output_dir}")


def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NASA Kepler Exoplanet Model Training')
    parser.add_argument('--train-data', required=True, help='Training data CSV file')
    parser.add_argument('--test-data', required=True, help='Test data CSV file')
    parser.add_argument('--output', default='models/results', help='Output directory')
    parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds')
    parser.add_argument('--tune-hyperparams', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--models', nargs='+', default=['lr', 'rf', 'xgb'], 
                       help='Models to train: lr, rf, xgb, lgb, nn')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    trainer.load_data(args.train_data, args.test_data)
    
    # Create models based on selection
    models = []
    model_map = {
        'lr': LogisticRegressionModel,
        'rf': RandomForestModel, 
        'xgb': XGBoostModel,
        'lgb': LightGBMModel,
        'nn': NeuralNetworkModel
    }
    
    for model_key in args.models:
        if model_key in model_map:
            try:
                models.append(model_map[model_key]())
            except ImportError as e:
                print(f"⚠️ Skipping {model_key}: {e}")
        else:
            print(f"⚠️ Unknown model: {model_key}")
    
    # Train models
    results = trainer.train_all_models(
        models, 
        cv_folds=args.cv_folds,
        hyperparameter_tuning=args.tune_hyperparams
    )
    
    # Print comparison
    print("\n📊 Model Comparison:")
    print("=" * 60)
    comparison_df = trainer.compare_models()
    print(comparison_df.to_string(index=False))
    
    # Get best model
    best_name, best_result = trainer.get_best_model()
    print(f"\n🏆 Best Model: {best_name}")
    print(f"   Accuracy: {best_result.mean_scores['accuracy']:.4f} ± {best_result.std_scores['accuracy']:.4f}")
    
    # Save results
    trainer.save_results(args.output)
    
    print(f"\n✅ Training completed! Results saved to: {args.output}")


if __name__ == "__main__":
    main()