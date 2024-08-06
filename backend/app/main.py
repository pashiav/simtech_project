import logging
import os
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.routes.prediction import router as prediction_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
# Mount the static files directory to serve the generated plot
app.mount("/temp_files", StaticFiles(directory="temp_files"), name="temp_files")

app.include_router(prediction_router, prefix="/predict")

@app.get("/")
def read_root():
    return {"message": "Lift Health Prediction API"}

@app.get("get_plot")
async def get_plot():
    return FileResponse("clustering_plot.png", media_type="image/png")