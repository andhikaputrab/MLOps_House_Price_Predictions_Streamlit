from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from typing import Dict, Any, Type
from src.utils.config import config
from src.utils.logger import default_logger as logger

class ModelFactory:
    
    @staticmethod
    def get_model_config() -> Dict[str, Dict[str, Any]]:
        """Get model configurations"""
        model_params = config.get('model_params', {})
        
        return {
            'linear_regression': {
                'class': LinearRegression,
                'params': {}
            },
            'random_forest_regressor': {
                'class': RandomForestRegressor,
                'params': model_params.get('random_forest_regressor', [])
            },
            'gradient_boosting_regressor': {
                'class': GradientBoostingRegressor,
                'params': model_params.get('gradient_boosting_regressor', [])
            }
        }
        
    @classmethod
    def create_model(cls, model_type: str) -> Any:
        """
        Create model instance
     
        Args:
            model_type: Type of model to create
         
        Returns:
            Model instance
        """
        try:
            logger.info(f"Creating model of type: {model_type}")

            # Get model configurations
            model_configs = cls.get_model_config()

            if model_type not in model_configs:
                raise ValueError(f"Unknown model type: {model_type}")

            # Get model class and parameters
            model_info = model_configs[model_type]
            model_class = model_info['class']
            model_params = model_info['params']

            # Create model instance
            model = model_class(**model_params)  # Instantiate the model

            logger.info(f"Successfully created {model_type} model")
            return model

        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise
        
class HousePriceModel:
    """Base class for house price prediction models"""

    def __init__(self, model_type: str):
        """
        Initialize house price model
        
        Args:
            model_type: Type of model to use
        """
        self.model_type = model_type
        self.model = None
        logger.info(f"Initialized HousePriceModel with type: {model_type}")

    def build(self) -> None:
        """Build model instance"""
        try:
            logger.info(f"Building {self.model_type} model")
            self.model = ModelFactory.create_model(self.model_type)
            logger.info("Model built successfully")
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
        
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if self.model is None:
            raise ValueError("Model not built yet")
        return self.model.get_params()
    
    def save_feature_importance(self, feature_names) -> Dict[str, float]:
        """
        Save feature importance
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        try:
            importance = self.get_feature_importance()
            if importance is not None:
                self.feature_importance = dict(zip(feature_names, importance))
                return self.feature_importance
            return None
        except Exception as e:
            logger.error(f"Error saving feature importance: {str(e)}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available"""
        if self.model is None:
            raise ValueError("Model not built yet")

        try:
            if hasattr(self.model, 'feature_importances_'):
                return self.model.feature_importances_
            else:
                logger.warning("Model doesn't support feature importance")
                return None
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            raise