from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Charger le modèle, le vectorizer et le MultiLabelBinarizer (si utilisé)
model = joblib.load("prediction_tags.pkl")
count_vectorizer = joblib.load("vectorizer_count.pkl")
mlb = joblib.load("multi_label_binarizer.pkl")  # Assure-toi d'avoir sauvegardé ceci lors de l'entraînement

# Définir l'API
app = FastAPI()

# Définir le format des requêtes
class InputData(BaseModel):
    titre: str
    texte: str

@app.post("/predict")
def predict(data: InputData):
    # Concaténer titre + texte
    texte_concatene = data.titre + " " + data.texte
    X_test = count_vectorizer.transform([texte_concatene])

    # Vérifier la cohérence des dimensions
    if X_test.shape[1] != model.n_features_in_:
        return {"error": f"Le modèle attend {model.n_features_in_} features, mais en a reçu {X_test.shape[1]}"}

    # Prédire la sortie vectorisée
    prediction_vect = model.predict(X_test)

    # Convertir la prédiction en texte
    prediction_labels = mlb.inverse_transform(prediction_vect)

    return {"prediction": prediction_labels}