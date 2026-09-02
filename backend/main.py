from fastapi import FastAPI

app = FastAPI(title="GeoFlow-Rescue API")


@app.get("/health")
def health_check():
    return {"status": "ok"}