import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
from typing import Tuple, Optional
from pathlib import Path
from src.utils.logger import default_logger as logger
from src.utils.config import config


class DataProcessor:
    """Data preprocessing pipeline"""

    def __init__(self, preprocessing_path: Optional[str] = None):
        """
        Initialize DataProcessor
        
        Args:
            preprocessing_path: Path to save/load preprocessing objects
        """
        self.preprocessing_path = preprocessing_path or config.get('preprocessing_path')
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.train_columns = None
        self.trained = False
        logger.info("Initialized DataProcessor")

    def _prepare_preprocessing_path(self) -> None:
        """Create preprocessing directory if it doesn't exist"""
        Path(self.preprocessing_path).mkdir(parents=True, exist_ok=True)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values for numerical and categorical columns"""
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in numerical_cols:
            df[col].fillna(df[col].median(), inplace=True)
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

        return df

    def _encode_categorical_columns(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical columns using LabelEncoder
        
        Args:
            df: Input DataFrame
            fit: Whether to fit new encoders or use existing ones
            
        Returns:
            Encoded DataFrame
        """
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                if col in self.label_encoders:
                    df[col] = self.label_encoders[col].transform(df[col])
                else:
                    raise ValueError(f"LabelEncoder for column '{col}' not found. Did you call fit_transform first?")
        return df

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
            df.drop(columns=config.get('columns_to_drop', []), inplace=True, errors='ignore')

            # Handle missing values
            df = self._handle_missing_values(df)

            # Filter important columns
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            corr_with_saleprice = df[numerical_cols].corr()["SalePrice"]
            important_num_cols = list(corr_with_saleprice[(corr_with_saleprice > 0.50) | (corr_with_saleprice < -0.50)].index)
            cat_cols = ["MSZoning", "Utilities", "BldgType", "Heating", "KitchenQual", "SaleCondition", "LandSlope"]

            # Combine important columns
            important_cols = important_num_cols + cat_cols

            # Process df (includes SalePrice)
            df = df[important_cols + ["SalePrice"]]

            # Encode categorical columns
            df = self._encode_categorical_columns(df, fit=True)
            
            # Scale numerical features
            logger.info("Fitting MinMaxScaler for numerical columns")
            numerical_cols = [col for col in important_num_cols if col in df.columns]
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])

            # Save train columns
            self.train_columns = df.drop(columns='SalePrice').columns.tolist()
            self._prepare_preprocessing_path()
            joblib.dump(self.train_columns, Path(self.preprocessing_path) / 'train_columns.joblib')
            # joblib.dump(self.label_encoders, Path(self.preprocessing_path) / 'label_encoders.joblib')

            # Remove duplicate columns if not done already
            df = df.loc[:, ~df.columns.duplicated()]

            # Split features and target
            X = df.drop(columns=[target_col], errors='ignore')
            y = df[target_col].values.ravel()

            self.trained = True
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
            df.drop(columns=config.get('columns_to_drop', []), inplace=True, errors='ignore')

            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Load train columns
            joblib_path = Path(self.preprocessing_path) / 'train_columns.joblib'
            if joblib_path.exists():
                self.train_columns = joblib.load(joblib_path)
            else:
                raise FileNotFoundError(f"Train columns file not found at {joblib_path}")

            # Load label encoders
            # encoder_path = Path(self.preprocessing_path) / 'label_encoders.joblib'
            # if encoder_path.exists():
            #     self.label_encoders = joblib.load(encoder_path)
            # else:
            #     raise FileNotFoundError(f"LabelEncoders file not found at {encoder_path}")

            # Encode categorical columns
            df = self._encode_categorical_columns(df, fit=False)

            # Align with train columns
            df = df.reindex(columns=self.train_columns, fill_value=0)

            # Scale numerical features
            numerical_cols = [col for col in self.train_columns if col in df.columns]
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
            joblib.dump(self.scaler, Path(self.preprocessing_path) / 'scaler.joblib')
            # joblib.dump(self.label_encoders, Path(self.preprocessing_path) / 'label_encoders.joblib')
            logger.info("Preprocessors saved successfully")
        except Exception as e:
            logger.error(f"Error saving preprocessors: {str(e)}")
            raise

    def load_preprocessors(self) -> None:
        """Load preprocessor objects"""
        try:
            logger.info(f"Loading preprocessors from {self.preprocessing_path}")
            self.scaler = joblib.load(Path(self.preprocessing_path) / 'scaler.joblib')
            # self.label_encoders = joblib.load(Path(self.preprocessing_path) / 'label_encoders.joblib')
            self.trained = True
            logger.info("Preprocessors loaded successfully")
        except Exception as e:
            logger.error(f"Error loading preprocessors: {str(e)}")
            raise
