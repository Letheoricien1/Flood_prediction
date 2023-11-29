from fastapi import FastAPI
from typing import List

app = FastAPI()

@app.get("/")
async def root():
 return {"greeting":"Flood prediction"}

@app.post("/analyse")
async def analyse_data(ville: str, valeurs: List[float]):
    x= len(valeurs)
    resultat = {
        "juillet": {"flood": True, "prob": 98},
        "aout": {"flood": True, "prob": 98},
        "septembre": {"flood": True, "prob": 98},
    }

    response = {"ville": ville, "resultat": resultat}
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)
