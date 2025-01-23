from fastapi import FastAPI, HTTPException
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os
import pickle
from datetime import datetime
from src.utils.logger import default_logger as logger
from src.data.data_processor import DataProcessor
from src.api.schemas import ModelInfo, ModelMetrics, HousePricePredictionRequest, HousePricePredictionResponse

app = FastAPI(
    title="House Prediction API",
    description="API for predicting house prices",
    version="1.0.0"
)

def get_latest_model_path() -> tuple[str, ModelInfo]:
    """
    Get the path to the latest trained model
    
    Returns:
        Tuple containing model path and model info
    """
    client = MlflowClient()
    
    experiment_name = "house_price_prediction_experiment"
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"No experiment found with name '{experiment_name}'")
    
    logger.info(f"Found experiment with ID: {experiment.experiment_id}")
    
    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.r2 DESC"]
    )

    if not runs:
        raise ValueError("No runs found in the experiment")
    
    # Find best run based on r2 score
    best_run = None
    best_score = -1
    
    for run in runs:
        metrics = run.data.metrics
        if 'r2' in metrics:
            new_best_score = metrics['r2']
            logger.info(f"Run {run.info.run_id} score: {new_best_score}")
            if new_best_score > best_score:
                best_score = new_best_score
                best_run = run
    
    if not best_run:
        raise ValueError("No valid runs found with required metrics")
    
    # best_run = runs[0]
    # run_id = best_run.info.run_id
    # logger.info(f"Best run ID: {run_id}")
    
    # Get model path
    run_id = best_run.info.run_id
    logger.info(f"Best run ID: {run_id}")
    
    # Try to load model by run ID
    try:
        # Load directly from run ID and model name
        model_path = f"runs:/{run_id}/gradient_boosting_regressor"
        logger.info(f"Trying to load model from: {model_path}")
        model = mlflow.pyfunc.load_model(model_path)
        logger.info(f"Successfully loaded model from: {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Try alternative path with 'model' artifact name
        try:
            model_path = f"runs:/{run_id}/model"
            logger.info(f"Trying alternative path: {model_path}")
            model = mlflow.pyfunc.load_model(model_path)
            logger.info("Successfully loaded model from alternative path")
        except Exception as e2:
            logger.error(f"Error loading from alternative path: {str(e2)}")
            # Try local filesystem path
            try:
                local_path = os.path.join("mlruns", experiment.experiment_id, 
                                        run_id, "artifacts/model")
                logger.info(f"Trying local filesystem path: {local_path}")
                if not os.path.exists(local_path):
                    raise ValueError(f"Local path does not exist: {local_path}")
                model = mlflow.pyfunc.load_model(local_path)
                model_path = local_path
                logger.info("Successfully loaded model from local path")
            except Exception as e3:
                logger.error(f"Error loading from local path: {str(e3)}")
                raise ValueError(f"Could not load model from any path. Errors:\n"
                               f"Primary: {str(e)}\n"
                               f"Alternative: {str(e2)}\n"
                               f"Local: {str(e3)}")
        
    metrics = ModelMetrics(
        rmse=best_run.data.metrics.get('rmse', 0.0),
        mae=best_run.data.metrics.get('mae', 0.0),
        r2=best_run.data.metrics.get('r2', 0.0)
    )
    
    model_info = ModelInfo(run_id=run_id, metrics=metrics)
    return model_path, model_info

@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor on startup"""
    global model, preprocessor, model_info
    try:
        logger.info("Loading best model from MLflow")

        # Set MLflow tracking URI
        mlflow.set_tracking_uri('sqlite:///mlflow.db')

        # Load the latest model
        # model_path, model_info = get_latest_model_path()
        # logger.info(f"Loading model from path: {model_path}")
        # model = mlflow.sklearn.load_model(model_path)
        
        model_path = 'mlruns/1/dd8b7ac4ca5b4e479f3a7b877034339c/artifacts/gradient_boosting_regressor/model.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Initialize and load preprocessor
        preprocessor = DataProcessor()
        preprocessor.load_preprocessors()

        logger.info("Model and preprocessor loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "House Price Prediction API",
        "timestamp": datetime.now().isoformat()
    }
    
@app.post("/predict", response_model=HousePricePredictionResponse)
async def predict(request: HousePricePredictionRequest):
    """
    Predict house price
    
    Args:
        request: Prediction request containing house features
        
    Returns:
        Prediction response with predicted house price and model info
    """
    try:
        logger.info("Received prediction request")

        # Convert request to DataFrame
        data = pd.DataFrame([request.dict()])

        # Preprocess data
        processed_data = preprocessor.transform(data)

        # Make prediction
        predicted_price = model.predict(processed_data)[0]

        response = HousePricePredictionResponse(
            predicted_price=float(predicted_price),
            model_info=model_info
        )
        logger.info("Prediction completed successfully")
        return response

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_info": model_info
    }