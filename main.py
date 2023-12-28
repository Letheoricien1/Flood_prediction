from fastapi import FastAPI
from typing import List
import pandas as pd
from joblib import dump, load
import pickle

model1 =pickle.load(open("model_Kaele_2_77.pkl", "rb"))

# Charger les modèle de Guider
with open('model_Guider_2_85.pkl', 'rb') as file:
    model_Guider_2_85= pickle.load(file)
with open('model_Guider_3_92.pkl', 'rb') as file:
    model_Guider_3_92= pickle.load(file)
with open('model_Guider_4_92.pkl', 'rb') as file:
    model_Guider_4_92= pickle.load(file)

# charger les modèles de Kaele
with open('model_Kaele_2_77.pkl', 'rb') as file:
    model_Kaele_2_77= pickle.load(file)
with open('model_Kaele_3_92.pkl', 'rb') as file:
    model_Kaele_3_92= pickle.load(file)
with open('model_Kaele_4_92.pkl', 'rb') as file:
    model_Kaele_4_92= pickle.load(file)

# charger les modèles de Tibati
with open('model_Tibati_2_70.pkl', 'rb') as file:
    model_Tibati_2_70 = pickle.load(file)
with open('model_Tibati_3_77.pkl', 'rb') as file:
    model_Tibati_3_77 = pickle.load(file)
with open('model_Tibati_4_85.pkl', 'rb') as file:
    model_Tibati_4_85 = pickle.load(file)
with open('model_Tibati_5_92.pkl', 'rb') as file:
    model_Tibati_5_92 = pickle.load(file)

# charger les modèles de Touboro
with open('model_Touboro_2_92.pkl', 'rb') as file:
    model_Touboro_2_92 = pickle.load(file)
with open('model_Touboro_3_92.pkl', 'rb') as file:
    model_Touboro_3_92 = pickle.load(file)
with open('model_Touboro_4_92.pkl', 'rb') as file:
    model_Touboro_4_92 = pickle.load(file)

# charger les modèles de Yagoua
with open('model_Yagoua_2_92.pkl', 'rb') as file:
    model_Yagoua_2_92 = pickle.load(file)
with open('model_Yagoua_3_84.pkl', 'rb') as file:
    model_Yagoua_3_92 = pickle.load(file)
with open('model_Yagoua_4_84.pkl', 'rb') as file:
    model_Yagoua_4_92 = pickle.load(file)

prob: int =0
label:int  =0
valeurs =[]

app = FastAPI()

@app.get("/")
async def root():
 return {"greeting":"Flood prediction f"}

@app.post("/analyse")
async def analyse_data(ville: str, valeurs: List[float]):
    global label, prob
    valeurs=list(valeurs)
    resultat = {
        "juillet": {"flood": None, "prob": None},
        "aout": {"flood": None, "prob": None},
        "septembre": {"flood": None, "prob": None},
    }
    if len(valeurs) ==2:
        df_pred=pd.DataFrame([valeurs], columns=["APR", "MAY"])
    if len(valeurs) ==3:
        df_pred=pd.DataFrame([valeurs], columns=["APR", "MAY", "JUN"])
    if len(valeurs) == 4:
        df_pred = pd.DataFrame([valeurs], columns=["APR", "MAY", "JUN", "JUL"])
    if len(valeurs) == 5:
        df_pred = pd.DataFrame([valeurs], columns=["MAR","APR", "MAY", "JUN", "JUL"])
    if ville=="Guider":
        if len(valeurs)==2:
            label=model_Guider_2_85.predict(df_pred)[0]
            prob=0.85
        if len(valeurs)==3:
            label=model_Guider_3_92.predict(df_pred)[0]
            prob=0.92
        if len(valeurs)==4:
            label=model_Guider_4_92.predict(df_pred)[0]
            prob =0.92

    if ville == "Kaele":
        if len(valeurs) == 2:
            #label = model_Kaele_2_77.predict(df_pred)
            label = model1.predict(df_pred)[0]
            prob = 0.77
        if len(valeurs) == 3:
            label = model_Kaele_3_92.predict(df_pred)[0]
            prob = 0.92
        if len(valeurs) == 4:
            label = model_Kaele_4_92.predict(df_pred)[0]
            prob = 0.92
    if ville == "Tibati":
        if len(valeurs) == 2:
            label = model_Tibati_2_70.predict(df_pred)[0]
            prob = 0.70
        if len(valeurs) == 3:
            label = model_Tibati_3_77.predict(df_pred)[0]
            prob = 0.77
        if len(valeurs) == 4:
            label = model_Tibati_4_85.predict(df_pred)[0]
            prob = 0.85
        if len(valeurs) == 5:
            label = model_Tibati_5_92.predict(df_pred)[0]
            prob = 0.92
    if ville == "Touboro":
        if len(valeurs) == 2:
            label = model_Touboro_2_92.predict(df_pred)[0]
            prob = 0.92
        if len(valeurs) == 3:
            label = model_Touboro_3_92.predict(df_pred)[0]
            prob = 0.92
        if len(valeurs) == 4:
            label = model_Touboro_4_92.predict(df_pred)[0]
            prob = 0.92
    if ville == "Yagoua":
        if len(valeurs) == 2:
            label = model_Kaele_2_77.predict(df_pred)[0]
            prob = 0.77
        if len(valeurs) == 3:
            label = model_Kaele_3_77.predict(df_pred)[0]
            prob = 0.77
        if len(valeurs) == 4:
            label = model_Kaele_4_92.predict(df_pred)[0]
            prob = 0.92

    x= len(valeurs)
    resultat["juillet"]["flood"] = int(label)
    resultat["juillet"]["prob"] = float(prob)
    resultat["aout"]["flood"] = int(label)
    resultat["aout"]["prob"] = float(prob)
    resultat["septembre"]["flood"] = int(label)
    resultat["septembre"]["prob"] = float(prob)
    if x>3:
        resultat["juillet"]["flood"] = -1
        resultat["juillet"]["prob"] =  -1

    response = {"ville": ville, "resultat": resultat}
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)
