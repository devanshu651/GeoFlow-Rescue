from pydantic import BaseModel, Field

class SimulationRequest(BaseModel):
    rainfall_mm: float = Field(..., ge=0.0, description="Total rainfall in millimeters")
    infiltration_rate: float = Field(..., ge=0.0, le=1.0, description="Infiltration rate (0.0 to 1.0)")
    slope_threshold: float = Field(30.0, ge=0.0, description="Slope threshold in degrees")
    rainfall_threshold: float = Field(50.0, ge=0.0, description="Rainfall threshold in mm")
    water_threshold: float = Field(0.05, ge=0.0, description="Water depth threshold in meters")
    dem_file_path: str | None = Field(None, description="Path to the DEM file (for development only)")

class SimulationResponse(BaseModel):
    status: str
    rainfall_mm: float
    max_water_depth: float
    flooded_cell_count: int
    first_pooling_cell_count: int
    flood_risk_max: float
    flood_risk_mean: float
    landslide_risk_max: float
    landslide_risk_mean: float
    high_risk_landslide_cell_count: int
