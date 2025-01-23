import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple, Dict, Optional
from pathlib import Path
from src.utils.logger import default_logger as logger
from src.utils.config import config

class DataProcessor:
    """Data preprocessing pipeline"""
    
    def __init__(self, preprocessing_path: Optional[str] = None):
        """
        Initialize data processor
        
        Args:
            preprocessing_path: Path to save/load preprocessing objects
        """
        self.preprocessing_path = preprocessing_path or config.get('preprocessing_path', 'models/preprocessing')
        self.scaler = MinMaxScaler()
        # self.train_columns = None
        self.trained = False
        logger.info("Initialized DataProcessor")
        
    def _prepare_preprocessing_path(self) -> None:
        """Create preprocessing directory if it doesn't exist"""
        Path(self.preprocessing_path).mkdir(parents=True, exist_ok=True)
        
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'SalePrice') -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fit preprocessors and transform data
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of transformed features and target
        """
        try:
            logger.info("Starting fit_transform process")
        
            # Drop irrelevant columns
            df.drop(columns=config.get('columns_to_drop', []), inplace=True)  # Drop columns with high missing values
        
            # Handle missing values for numerical and categorical columns
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = df.select_dtypes(include=['object']).columns

            # Impute missing values
            for col in numerical_cols:
                if col in df.columns:
                    df[col].fillna(df[col].median(), inplace=True)
            for col in categorical_cols:
                if col in df.columns:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    
            corr_with_saleprice = df[numerical_cols].corr()["SalePrice"]
            important_num_cols = list(corr_with_saleprice[(corr_with_saleprice > 0.50) | (corr_with_saleprice < -0.50)].index)
            cat_cols = ["MSZoning", "Utilities","BldgType","Heating","KitchenQual","SaleCondition","LandSlope"]
            important_cols = important_num_cols + cat_cols
            df = df[important_cols]
        
            # Encode categorical columns
            df = pd.get_dummies(df, drop_first=True)

            # Scale numerical features
            logger.info("Fitting MinMaxScaler for numerical columns")
            numerical_cols = [col for col in important_num_cols if col in df.columns]
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])

            # Split features and target
            X = df.drop(columns=[target_col], errors='ignore')
            y = df[target_col] if target_col in df.columns else None
        
            # Save preprocessors
            self.save_preprocessors()
            
            logger.info("Fit_transform completed successfully")
            return X, y

        except Exception as e:
            logger.error(f"Error in fit_transform: {str(e)}")
            raise
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessors
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self.trained:
            raise ValueError("DataProcessor not fitted. Call fit_transform first.")
            
        try:
            logger.info("Starting transform process")
            
            # Drop irrelevant columns
            df.drop(columns=config.get('columns_to_drop', []), inplace=True)  # Drop columns with high missing values
        
            # Handle missing values for numerical and categorical columns
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = df.select_dtypes(include=['object']).columns

            # Impute missing values
            for col in numerical_cols:
                if col in df.columns:
                    df[col].fillna(df[col].median(), inplace=True)
            for col in categorical_cols:
                if col in df.columns:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    
            corr_with_saleprice = df[numerical_cols].corr()["SalePrice"]
            important_num_cols = list(corr_with_saleprice[(corr_with_saleprice > 0.50) | (corr_with_saleprice < -0.50)].index)
            cat_cols = ["MSZoning", "Utilities","BldgType","Heating","KitchenQual","SaleCondition","LandSlope"]
            important_cols = important_num_cols + cat_cols
            df = df[important_cols]
        
            # Encode categorical columns
            df = pd.get_dummies(df, drop_first=True)
            
            # Align the train and test datasets to ensure they have the same columns
            # train_cols_path = Path(self.preprocessing_path) / 'train_columns.joblib'
            # self.train_columns = joblib.load(train_cols_path)

            # df = df.reindex(columns=self.train_columns, fill_value=0)

            # Scale numerical features
            # df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            logger.info("Fitting MinMaxScaler for numerical columns")
            numerical_cols = [col for col in important_num_cols if col in df.columns]
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])

            logger.info("Transform completed successfully")
            return df

        except Exception as e:
            logger.error(f"Error in transform: {str(e)}")
            raise
        
    def save_preprocessors(self) -> None:
        """Save preprocessor objects and train columns"""
        try:
            logger.info(f"Saving preprocessors to {self.preprocessing_path}")
            self._prepare_preprocessing_path()

            # Save scaler
            joblib.dump(self.scaler, Path(self.preprocessing_path) / 'scaler.joblib')

            # Save train columns
            # joblib.dump(self.train_columns, Path(self.preprocessing_path) / 'train_columns.joblib')

            logger.info("Preprocessors saved successfully")

        except Exception as e:
            logger.error(f"Error saving preprocessors: {str(e)}")
            raise
  
    def load_preprocessors(self) -> None:
        """Load preprocessor objects"""
        try:
            logger.info(f"Loading preprocessors from {self.preprocessing_path}")

            # Load scaler
            scaler_path = Path(self.preprocessing_path) / 'scaler.joblib'
            self.scaler = joblib.load(scaler_path)

            self.trained = True
            logger.info("Preprocessors loaded successfully")

        except Exception as e:
            logger.error(f"Error loading preprocessors: {str(e)}")
            raise