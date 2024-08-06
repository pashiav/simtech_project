import os
import pandas as pd
import zipfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from app.models.lift_anomaly_model import LiftAnomalyModel
import logging

router = APIRouter()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
@router.post("/")
async def predict(files: List[UploadFile] = File(...)):
    try:
        # Ensure temp directory exists
        temp_dir = "temp_files"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Extract files
        file_paths = []
        for file in files:
            contents = await file.read()
            file_path = f"{temp_dir}/{file.filename}"
            with open(file_path, "wb") as f:
                f.write(contents)
            file_paths.append(file_path)
        
        logger.info(f"Uploaded files: {file_paths}")

        # Unzip files
        extracted_files = []
        for file_path in file_paths:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                extracted_files.extend(zip_ref.namelist())
        
        logger.info(f"Extracted files: {extracted_files}")

        # Filter out __MACOSX files and only keep CSV files
        valid_files = [f for f in extracted_files if not f.startswith('__MACOSX') and f.endswith('.csv')]
        
        logger.info(f"Valid files: {valid_files}")

        if len(valid_files) == 0:
            raise HTTPException(status_code=400, detail="No valid CSV files found in the ZIP.")
        
        # Create full paths for valid files
        valid_file_paths = [os.path.join(temp_dir, f) for f in valid_files]
        
        logger.info(f"Valid file paths: {valid_file_paths}")

        # Load and predict using the model
        model = LiftAnomalyModel()
        results = model.predict(valid_file_paths)

        model.generate_plot()
        
        # Clean up temp files
        for file_path in file_paths:
            os.remove(file_path)
        for valid_file_path in valid_file_paths:
            os.remove(valid_file_path)
        
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
