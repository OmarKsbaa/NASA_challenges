#!/usr/bin/env python3
"""
NASA Kepler Exoplanet Data Preprocessing Pipeline

This script preprocesses the NASA Kepler cumulative exoplanet dataset for machine learning.
It handles missing values, applies transformations, engineers features, and splits data.

Usage:
    python src/preprocess.py --input "cumulative_2025.09.25_10.52.58.csv" --output data/processed --test-size 0.2 --seed 42

Author: NASA Space Apps Challenge Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import argparse
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import warnings
warnings.filterwarnings('ignore')

class KeplerPreprocessor:
    """
    Comprehensive preprocessing pipeline for NASA Kepler exoplanet data
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_map = {'FALSE POSITIVE': 0, 'CANDIDATE': 1, 'CONFIRMED': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        self.scaler = None
        self.imputer_simple = None
        self.imputer_iterative = None
        self.feature_columns = []
        
    def load_data(self, file_path):
        """Load and validate the Kepler dataset"""
        print(f"📂 Loading data from: {file_path}")
        
        try:
            # Read CSV with comment handling
            df = pd.read_csv(file_path, comment='#', low_memory=False)
            print(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Validate required columns
            required_cols = ['koi_pdisposition']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def encode_target(self, df):
        """Encode target variable to integers"""
        print("🎯 Encoding target variable...")
        
        df = df.copy()
        df['target'] = df['koi_pdisposition'].map(self.label_map)
        
        # Check for unmapped values
        unmapped = df['target'].isnull().sum()
        if unmapped > 0:
            print(f"⚠️  {unmapped} rows with unmapped target values")
            # Fill unmapped with most common class (FALSE POSITIVE = 0)
            df['target'] = df['target'].fillna(0)
        
        print(f"Target distribution:")
        for label, count in df['target'].value_counts().sort_index().items():
            class_name = self.reverse_label_map[label]
            pct = count / len(df) * 100
            print(f"  {label} ({class_name}): {count:,} ({pct:.1f}%)")
            
        return df
    
    def select_columns(self, df):
        """Select and categorize columns for preprocessing"""
        print("📋 Selecting columns for preprocessing...")
        
        # Define column categories based on README specification
        drop_cols = [
            'loc_rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition'
        ]
        
        # Core features for modeling
        core_features = [
            'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 
            'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff', 
            'koi_slogg', 'koi_srad', 'koi_kepmag'
        ]
        
        # Binary flags
        flag_features = [
            'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
        ]
        
        # Error/uncertainty columns for feature engineering
        error_cols = [
            'koi_period_err1', 'koi_period_err2', 'koi_depth_err1', 'koi_depth_err2',
            'koi_duration_err1', 'koi_duration_err2', 'koi_prad_err1', 'koi_prad_err2'
        ]
        
        # Optional features
        optional_features = ['koi_score', 'ra', 'dec', 'koi_impact']
        
        # Keep columns that exist in the dataset
        available_cols = set(df.columns)
        keep_features = []
        
        for col_list, name in [(core_features, "core"), (flag_features, "flags"), 
                              (optional_features, "optional")]:
            existing = [col for col in col_list if col in available_cols]
            keep_features.extend(existing)
            print(f"  {name.capitalize()} features: {len(existing)}/{len(col_list)} available")
        
        # Add error columns for engineering
        error_available = [col for col in error_cols if col in available_cols]
        keep_features.extend(error_available)
        print(f"  Error columns: {len(error_available)}/{len(error_cols)} available")
        
        # Remove duplicates and sort
        self.feature_columns = sorted(list(set(keep_features)))
        print(f"  Total features selected: {len(self.feature_columns)}")
        
        # Drop unnecessary columns
        df_processed = df[['target', 'koi_pdisposition'] + self.feature_columns].copy()
        
        return df_processed
    
    def engineer_features(self, df):
        """Create engineered features"""
        print("🔧 Engineering features...")
        
        df = df.copy()
        
        # 1. Log transformations for skewed physical quantities
        log_features = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol']
        for col in log_features:
            if col in df.columns:
                # Handle zeros and negatives by adding small constant
                df[f'log_{col}'] = np.log10(df[col] + 1e-8)
                print(f"  ✅ Created log_{col}")
        
        # 2. Uncertainty features (combine err1 and err2)
        uncertainty_pairs = [
            ('koi_period_err1', 'koi_period_err2', 'period_err_abs'),
            ('koi_depth_err1', 'koi_depth_err2', 'depth_err_abs'),
            ('koi_duration_err1', 'koi_duration_err2', 'duration_err_abs'),
            ('koi_prad_err1', 'koi_prad_err2', 'prad_err_abs')
        ]
        
        for err1, err2, new_col in uncertainty_pairs:
            if err1 in df.columns and err2 in df.columns:
                df[new_col] = np.maximum(np.abs(df[err1]), np.abs(df[err2]))
                print(f"  ✅ Created {new_col}")
        
        # 3. Ratio features
        if 'koi_depth' in df.columns and 'koi_model_snr' in df.columns:
            df['snr_depth_ratio'] = df['koi_depth'] / (df['koi_model_snr'] + 1e-8)
            print(f"  ✅ Created snr_depth_ratio")
            
        if 'koi_period' in df.columns and 'koi_duration' in df.columns:
            df['period_duration_ratio'] = df['koi_period'] / (df['koi_duration'] + 1e-8)
            print(f"  ✅ Created period_duration_ratio")
        
        # 4. Missing value flags for key features
        key_features = ['koi_period', 'koi_depth', 'koi_prad', 'koi_steff']
        for col in key_features:
            if col in df.columns:
                df[f'{col}_missing'] = df[col].isnull().astype(int)
                print(f"  ✅ Created {col}_missing flag")
        
        return df
    
    def handle_missing_values(self, df, mode='train'):
        """
        Handle missing values with domain-aware astronomical imputation strategies
        
        Strategy Overview:
        1. Drop completely missing error columns (koi_teq_err1/err2: 100% missing)
        2. Handle kepler_name strategically (71% missing - normal for candidates)  
        3. Use physics-informed imputation for stellar/planetary parameters
        4. Apply iterative imputation for correlated measurements
        5. Conservative median for robust core features
        """
        print(f"🔄 Handling missing values with astronomy-aware strategies (mode: {mode})...")
        
        df = df.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target and non-feature columns from processing
        feature_numeric_cols = [col for col in numeric_cols 
                               if col not in ['target'] and not col.endswith('_missing')]
        feature_categorical_cols = [col for col in categorical_cols 
                                   if col not in ['target', 'koi_pdisposition']]
        
        # Calculate missing percentages for analysis
        if feature_numeric_cols:
            missing_pct = df[feature_numeric_cols].isnull().mean() * 100
            print(f"  📊 Missing value analysis: {len(feature_numeric_cols)} numeric features")
        else:
            print("  No numeric features to analyze")
            return df
        
        # === STRATEGY 1: Drop completely missing error columns (100% missing) ===
        completely_missing_cols = missing_pct[missing_pct >= 99.9].index.tolist()
        if completely_missing_cols:
            df = df.drop(columns=completely_missing_cols)
            feature_numeric_cols = [col for col in feature_numeric_cols if col not in completely_missing_cols]
            print(f"  🗑️  Dropped completely missing columns: {completely_missing_cols}")
        
        # === STRATEGY 2: Handle kepler_name (identifier, high missingness is normal) ===
        if 'kepler_name' in feature_categorical_cols:
            # For confirmed planets, kepler_name exists; for candidates, it's often missing
            # This is informative - create a binary feature instead of imputing
            df['has_kepler_name'] = (~df['kepler_name'].isnull()).astype(int)
            df = df.drop(columns=['kepler_name'])  # Remove original after creating feature
            print(f"  🏷️  Converted kepler_name to binary feature 'has_kepler_name'")
        
        # Recalculate missing percentages after dropping columns
        if feature_numeric_cols:
            missing_pct = df[feature_numeric_cols].isnull().mean() * 100
        
        # === STRATEGY 3: Physics-informed imputation for stellar parameters ===
        # Group related measurements that should be imputed together
        stellar_params = [col for col in feature_numeric_cols if any(x in col.lower() for x in 
                         ['koi_srad', 'koi_smass', 'koi_sage', 'koi_slogg', 'koi_steff'])]
        planetary_params = [col for col in feature_numeric_cols if any(x in col.lower() for x in 
                           ['koi_prad', 'koi_period', 'koi_dor', 'koi_incl', 'koi_ror'])]
        transit_params = [col for col in feature_numeric_cols if any(x in col.lower() for x in 
                         ['koi_duration', 'koi_depth', 'koi_ingress', 'koi_egress'])]
        
        # === STRATEGY 4: Conservative median imputation for low missingness (<5%) ===
        very_low_missing_cols = missing_pct[missing_pct <= 5.0].index.tolist()
        if very_low_missing_cols and mode == 'train':
            self.imputer_simple = SimpleImputer(strategy='median')
            df[very_low_missing_cols] = self.imputer_simple.fit_transform(df[very_low_missing_cols])
            print(f"  📊 Conservative median imputation (<5% missing): {len(very_low_missing_cols)} columns")
        elif very_low_missing_cols and mode == 'predict' and hasattr(self, 'imputer_simple'):
            df[very_low_missing_cols] = self.imputer_simple.transform(df[very_low_missing_cols])
            print(f"  📊 Applied median imputation: {len(very_low_missing_cols)} columns")
        
        # === STRATEGY 5: Iterative imputation for correlated parameter groups (5-30%) ===
        moderate_missing_cols = missing_pct[(missing_pct > 5.0) & (missing_pct <= 30.0)].index.tolist()
        if moderate_missing_cols and mode == 'train':
            # Create groups for physics-aware imputation
            imputation_groups = []
            
            # Group 1: Stellar parameters (often correlated)
            stellar_moderate = [col for col in moderate_missing_cols if col in stellar_params]
            if stellar_moderate:
                # Include some low-missing stellar params for better imputation
                stellar_context = stellar_moderate + [col for col in very_low_missing_cols if col in stellar_params][:3]
                imputation_groups.append(('stellar', stellar_moderate, stellar_context))
            
            # Group 2: Planetary parameters (physically related)
            planetary_moderate = [col for col in moderate_missing_cols if col in planetary_params]
            if planetary_moderate:
                planetary_context = planetary_moderate + [col for col in very_low_missing_cols if col in planetary_params][:3]
                imputation_groups.append(('planetary', planetary_moderate, planetary_context))
            
            # Group 3: Transit parameters (observational, often correlated)
            transit_moderate = [col for col in moderate_missing_cols if col in transit_params]
            if transit_moderate:
                transit_context = transit_moderate + [col for col in very_low_missing_cols if col in transit_params][:3]
                imputation_groups.append(('transit', transit_moderate, transit_context))
            
            # Group 4: Remaining moderate missing columns (general approach)
            remaining_moderate = [col for col in moderate_missing_cols 
                                if col not in stellar_moderate + planetary_moderate + transit_moderate]
            if remaining_moderate:
                # Use most complete features as context
                general_context = remaining_moderate + very_low_missing_cols[:5]
                imputation_groups.append(('general', remaining_moderate, general_context))
            
            # Apply iterative imputation to each group
            self.imputers_iterative = {}
            for group_name, target_cols, context_cols in imputation_groups:
                if len(context_cols) >= 2:  # Need at least 2 features for iterative imputation
                    imputer = IterativeImputer(
                        max_iter=10, 
                        random_state=self.random_state,
                        initial_strategy='median'  # More robust initial guess
                    )
                    # Fit on context columns, transform target columns
                    imputed_data = imputer.fit_transform(df[context_cols])
                    target_indices = [context_cols.index(col) for col in target_cols if col in context_cols]
                    df[target_cols] = imputed_data[:, target_indices]
                    self.imputers_iterative[group_name] = (imputer, context_cols, target_cols)
                    print(f"  🧮 Iterative imputation ({group_name}): {len(target_cols)} columns")
                else:
                    # Fallback to median for small groups
                    df[target_cols] = df[target_cols].fillna(df[target_cols].median())
                    print(f"  📊 Median fallback ({group_name}): {len(target_cols)} columns")
        
        elif moderate_missing_cols and mode == 'predict' and hasattr(self, 'imputers_iterative'):
            # Apply trained iterative imputers
            for group_name, (imputer, context_cols, target_cols) in self.imputers_iterative.items():
                available_context = [col for col in context_cols if col in df.columns]
                available_targets = [col for col in target_cols if col in df.columns]
                if len(available_context) >= 2 and available_targets:
                    imputed_data = imputer.transform(df[available_context])
                    target_indices = [available_context.index(col) for col in available_targets if col in available_context]
                    df[available_targets] = imputed_data[:, target_indices]
                    print(f"  🧮 Applied iterative imputation ({group_name}): {len(available_targets)} columns")
        
        # === STRATEGY 6: Handle high missingness columns (30-70%) ===
        high_missing_cols = missing_pct[(missing_pct > 30.0) & (missing_pct <= 70.0)].index.tolist()
        if high_missing_cols:
            # For astronomy data, some high missingness might be informative
            # Create missingness indicators and use simple imputation
            for col in high_missing_cols:
                df[f'{col}_was_missing'] = df[col].isnull().astype(int)
            
            # Simple median imputation for high missing (conservative approach)
            df[high_missing_cols] = df[high_missing_cols].fillna(df[high_missing_cols].median())
            print(f"  ⚠️  High missingness (30-70%) with indicators: {len(high_missing_cols)} columns")
        
        # === STRATEGY 7: Drop extremely high missingness columns (>70%) ===
        extreme_missing_cols = missing_pct[missing_pct > 70.0].index.tolist()
        if extreme_missing_cols:
            df = df.drop(columns=extreme_missing_cols)
            print(f"  🗑️  Dropped extreme missingness (>70%): {extreme_missing_cols}")
        
        # Final validation - ensure no remaining missing values in numeric columns
        remaining_numeric = [col for col in df.select_dtypes(include=[np.number]).columns 
                           if col not in ['target'] and not col.endswith('_missing') and not col.endswith('_was_missing')]
        remaining_missing = df[remaining_numeric].isnull().sum()
        if remaining_missing.sum() > 0:
            print(f"  🔧 Final cleanup: {remaining_missing.sum()} remaining missing values")
            df[remaining_numeric] = df[remaining_numeric].fillna(df[remaining_numeric].median())
        
        print(f"  ✅ Missing value handling complete - astronomy-aware strategy applied")
        return df
    
    def scale_features(self, df, mode='train'):
        """Scale features using RobustScaler"""
        print(f"📏 Scaling features (mode: {mode})...")
        
        df = df.copy()
        
        # Get numeric columns (excluding target and flags)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        scale_cols = [col for col in numeric_cols 
                     if col not in ['target'] and not col.endswith('_missing') 
                     and not col.startswith('koi_fpflag')]
        
        if len(scale_cols) == 0:
            print("  No columns to scale")
            return df
        
        if mode == 'train':
            self.scaler = RobustScaler()
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
            print(f"  ✅ Fitted and scaled {len(scale_cols)} features")
        elif mode == 'predict' and self.scaler:
            df[scale_cols] = self.scaler.transform(df[scale_cols])
            print(f"  ✅ Applied scaling to {len(scale_cols)} features")
        else:
            print("  ⚠️  No fitted scaler available")
        
        return df
    
    def split_data(self, df, test_size=0.2):
        """Split data into train and test sets with stratification"""
        print(f"✂️  Splitting data ({test_size:.0%} test size)...")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['target', 'koi_pdisposition']]
        X = df[feature_cols]
        y = df['target']
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        print(f"  Training set: {len(X_train):,} samples")
        print(f"  Test set: {len(X_test):,} samples")
        
        # Check stratification
        print("  Class distribution in splits:")
        for split_name, y_split in [("Train", y_train), ("Test", y_test)]:
            dist = y_split.value_counts(normalize=True).sort_index()
            print(f"    {split_name}: " + " | ".join([f"{self.reverse_label_map[k]}: {v:.1%}" 
                                                     for k, v in dist.items()]))
        
        return X_train, X_test, y_train, y_test
    
    def save_artifacts(self, output_dir, X_train, X_test, y_train, y_test):
        """Save processed data and preprocessing artifacts"""
        print(f"💾 Saving artifacts to: {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        artifacts_path = Path("artifacts")
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Save processed datasets
        train_df = X_train.copy()
        train_df['target'] = y_train.values
        test_df = X_test.copy()
        test_df['target'] = y_test.values
        
        train_file = output_path / "clean_train.csv"
        test_file = output_path / "clean_test.csv"
        
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        print(f"  ✅ Saved: {train_file}")
        print(f"  ✅ Saved: {test_file}")
        
        # Save preprocessing artifacts
        # Label mapping
        label_map_file = artifacts_path / "label_map.json"
        with open(label_map_file, 'w') as f:
            json.dump(self.label_map, f, indent=2)
        print(f"  ✅ Saved: {label_map_file}")
        
        # Scaler
        if self.scaler:
            scaler_file = artifacts_path / "scaler.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"  ✅ Saved: {scaler_file}")
        
        # Preprocessing config
        config = {
            'feature_columns': list(X_train.columns),
            'target_classes': self.reverse_label_map,
            'train_shape': train_df.shape,
            'test_shape': test_df.shape,
            'preprocessing_steps': [
                'target_encoding', 'feature_selection', 'feature_engineering',
                'missing_value_imputation', 'feature_scaling', 'train_test_split'
            ]
        }
        
        config_file = artifacts_path / "preprocess_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  ✅ Saved: {config_file}")
        
        return {
            'train_file': str(train_file),
            'test_file': str(test_file),
            'label_map_file': str(label_map_file),
            'scaler_file': str(scaler_file) if self.scaler else None,
            'config_file': str(config_file)
        }

def main():
    """Main preprocessing pipeline"""
    parser = argparse.ArgumentParser(description='NASA Kepler Exoplanet Data Preprocessing')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', default='data/processed', help='Output directory')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("🚀 NASA Kepler Exoplanet Data Preprocessing Pipeline")
    print("=" * 60)
    
    try:
        # Initialize preprocessor
        preprocessor = KeplerPreprocessor(random_state=args.seed)
        
        # Load data
        df = preprocessor.load_data(args.input)
        
        # Preprocessing pipeline
        df = preprocessor.encode_target(df)
        df = preprocessor.select_columns(df)
        df = preprocessor.engineer_features(df)
        df = preprocessor.handle_missing_values(df, mode='train')
        df = preprocessor.scale_features(df, mode='train')
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(df, test_size=args.test_size)
        
        # Save artifacts
        artifacts = preprocessor.save_artifacts(args.output, X_train, X_test, y_train, y_test)
        
        print(f"\n✅ Preprocessing completed successfully!")
        print(f"📊 Final dataset: {len(X_train.columns)} features")
        print(f"🎯 Target classes: {len(preprocessor.label_map)}")
        print(f"📁 Files saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()