from fastapi import FastAPI
from app.api.routes.simulation import router as simulation_router

app = FastAPI(title="GeoFlow-Rescue API")

app.include_router(simulation_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "ok"}
