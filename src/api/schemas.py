from pydantic import BaseModel, Field
from typing import Dict, Optional

class ModelMetrics(BaseModel):
    """Model metrics schema"""
    rmse: float = Field(..., description="Root Mean Squared Error of the model")
    mae: float = Field(..., description="Mean Absolute Error of the model")
    r2: float = Field(..., description="R² score of the model")

class ModelInfo(BaseModel):
    """Model information schema"""
    run_id: str = Field(..., description="MLflow run ID")
    metrics: ModelMetrics = Field(..., description="Model metrics")
    
class HousePricePredictionRequest(BaseModel):
    """House price prediction request schema"""
    OverallQual: int = Field(..., ge=1, le=10, description="Overall quality (1-10)")
    YearBuilt: int = Field(..., ge=1800, le=2025, description="Year the house was built")
    YearRemodAdd: int = Field(..., ge=1800, le=2025, description="Year the house was remodeled")
    TotalBsmtSF: float = Field(..., ge=0, description="Total basement square footage")
    FirstFlrSF: float = Field(..., ge=0, description="First floor square footage")
    GrLivArea: float = Field(..., ge=0, description="Above ground living area in square feet")
    FullBath: int = Field(..., ge=0, description="Number of full bathrooms")
    TotRmsAbvGrd: int = Field(..., ge=0, description="Total rooms above ground")
    GarageCars: int = Field(..., ge=0, description="Number of cars that can fit in the garage")
    GarageArea: float = Field(..., ge=0, description="Size of the garage in square feet")
    MSZoning: str = Field(..., ge=0, description="The general zoning classification")
    Utilities: str = Field(..., ge=0, description="Type of utilities available")
    BldgType: str = Field(..., ge=0, description="Type of dwelling")
    Heating: str = Field(..., ge=0, description="Type of heating")
    KitchenQual: str = Field(..., ge=0, description="Kitchen quality")
    SaleCondition: str = Field(..., ge=0, description="Condition of sale")
    LandSlope: str = Field(..., ge=0, description="Slope of property")
    

    # # Categorical features as boolean flags for one-hot encoding
    # MSZoning_FV: bool = Field(..., description="Is the zoning FV?")
    # MSZoning_RH: bool = Field(..., description="Is the zoning RH?")
    # MSZoning_RL: bool = Field(..., description="Is the zoning RL?")
    # MSZoning_RM: bool = Field(..., description="Is the zoning RM?")
    # Utilities_NoSeWa: bool = Field(..., description="Is the utility 'NoSeWa'?")
    # BldgType_2fmCon: bool = Field(..., description="Is the building type 2fmCon?")
    # BldgType_Duplex: bool = Field(..., description="Is the building type Duplex?")
    # BldgType_Twnhs: bool = Field(..., description="Is the building type Townhouse?")
    # BldgType_TwnhsE: bool = Field(..., description="Is the building type Townhouse End?")
    # Heating_GasA: bool = Field(..., description="Is the heating type GasA?")
    # Heating_GasW: bool = Field(..., description="Is the heating type GasW?")
    # Heating_Grav: bool = Field(..., description="Is the heating type Grav?")
    # Heating_OthW: bool = Field(..., description="Is the heating type Other?")
    # Heating_Wall: bool = Field(..., description="Is the heating type Wall?")
    # KitchenQual_Fa: bool = Field(..., description="Is the kitchen quality Fair?")
    # KitchenQual_Gd: bool = Field(..., description="Is the kitchen quality Good?")
    # KitchenQual_TA: bool = Field(..., description="Is the kitchen quality Typical/Average?")
    # SaleCondition_AdjLand: bool = Field(..., description="Is the sale condition Adjacent Land?")
    # SaleCondition_Alloca: bool = Field(..., description="Is the sale condition Allocation?")
    # SaleCondition_Family: bool = Field(..., description="Is the sale condition Family?")
    # SaleCondition_Normal: bool = Field(..., description="Is the sale condition Normal?")
    # SaleCondition_Partial: bool = Field(..., description="Is the sale condition Partial?")
    # LandSlope_Mod: bool = Field(..., description="Is the land slope Moderate?")
    # LandSlope_Sev: bool = Field(..., description="Is the land slope Severe?")

    class Config:
        schema_extra = {
            "example": {
                "OverallQual": 7,
                "YearBuilt": 2003,
                "YearRemodAdd": 2004,
                "TotalBsmtSF": 856.0,
                "FirstFlrSF": 856.0,
                "GrLivArea": 1710.0,
                "FullBath": 2,
                "TotRmsAbvGrd": 8,
                "GarageCars": 2,
                "GarageArea": 548.0,
                "MSZoning": "RL",
                "Utilities": "AllPub",
                "BldgType": "1Fam",
                "Heating": "GasA",
                "KitchenQual": "Gd",
                "SaleCondition": "Normal",
                "LandSlope": "Gtl"
            }
        }

class HousePricePredictionResponse(BaseModel):
    """House price prediction response schema"""
    predicted_price: float = Field(..., ge=0, description="Predicted house price")
    model_info: ModelInfo = Field(..., description="Model information and metrics")