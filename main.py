from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = FastAPI()

# Clustering centroids from INSPIRED study (mocked)
cluster_centroids = [
    [42.5, 24.9, 90, 10.7, 149, 40, 0.8],  # SIDD
    [46.5, 32.6, 108, 8.3, 155, 38, 1.5],  # IROD
    [42.1, 26.5, 94.9, 9.1, 351, 36, 1.2], # CIRDD
    [50.2, 25.9, 92.4, 7.2, 136, 42, 1.1]  # MARD
]

scaler = StandardScaler()
scaler.fit(cluster_centroids)
scaled_centroids = scaler.transform(cluster_centroids)
model = KMeans(n_clusters=4, init=scaled_centroids, n_init=1)
model.fit(scaled_centroids)

class InputData(BaseModel):
    age: float
    bmi: float
    waist: float
    hba1c: float
    tg: float
    hdl: float
    c_peptide: float

@app.post("/classify")
async def classify(data: InputData):
    try:
        features = np.array([[
            data.age, data.bmi, data.waist,
            data.hba1c, data.tg, data.hdl, data.c_peptide
        ]])
        scaled = scaler.transform(features)
        label = model.predict(scaled)[0]
        cluster_names = [
            "Severe Insulin Deficient Diabetes (SIDD)",
            "Insulin Resistant Obese Diabetes (IROD)",
            "Combined Insulin Resistant and Deficient Diabetes (CIRDD)",
            "Mild Age-Related Diabetes (MARD)"
        ]
        return { "cluster": cluster_names[label] }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
