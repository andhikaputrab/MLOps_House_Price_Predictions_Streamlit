import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from src.utils.logger import default_logger as logger
from src.utils.config import config

class DataLoader:
    
    def __init__(self, train_data_path: Optional[str] = None, test_data_path: Optional[str] = None):
        """
        initilize data loader

        Args:
            data_path: Optional path to data file
        """
        self.train_data_path = train_data_path or config.get('train_data_path')
        self.test_data_path = test_data_path or config.get('test_data_path')
        logger.info(f"initialized DataLoader with path: {self.train_data_path} and {self.test_data_path}")
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from file
        
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            logger.info(f"loading data from {self.train_data_path} and {self.test_data_path}")
            df_train = pd.read_csv(self.train_data_path)
            df_test = pd.read_csv(self.test_data_path)
            logger.info(f"loaded train data successfully with shape {df_train.shape}")
            logger.info(f"loaded test data successfully with shape {df_test.shape}")
            return df_train, df_test
        except Exception as e:
            logger.info(f"Error loading data: {e}")
            raise
            
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the loaded data
        
        Args:
            df (pd.DataFrame): DataFrame to validate
        
        Returns:
            bool: True if validation passes
        """
        try:
            logger.info("Validating data")
            
            # Check for required columns
            required_columns = config.get("required_columns", [])
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check for null values
            null_counts = df.isnull().sum()
            if null_counts.any():
                logger.warning(f"Found null values:\n{null_counts[null_counts > 0]}")
            
            logger.info("Data validation completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False
        
    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split data into features and target
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)
        """
        try:
            logger.info("Splitting features and target")
            target_column = config.get("target_column", "SalePrice")
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' is missing in the dataset")
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            logger.info(f"Split completed. Features shape: {X.shape}, Target shape: {y.shape}")
            return X, y
        except Exception as e:
            logger.error(f"Error splitting features and target: {e}")
            raise