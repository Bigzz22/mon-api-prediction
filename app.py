from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import re
import gensim
from bs4 import BeautifulSoup
from gensim.parsing.preprocessing import STOPWORDS
import spacy

# Charger le modèle de langue de spaCy
nlp = spacy.load("en_core_web_sm")  # ou "fr_core_news_sm" si ton texte est en français

def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()  # Supprimer HTML
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())  # Supprimer ponctuation et mettre en minuscule
    tokens = gensim.utils.simple_preprocess(text, deacc=True)  # Tokenization
    tokens = [word for word in tokens if word not in STOPWORDS]  # Supprimer stopwords
    tokens = [token.lemma_ for token in nlp(" ".join(tokens)) if token.pos_ in ["NOUN", "VERB"]]  # Lemmatisation
    return " ".join(tokens)

# Charger le modèle, le vectorizer et le MultiLabelBinarizer (si utilisé)
model = joblib.load("prediction_tags.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
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
    texte_pretraite = preprocess_text(texte_concatene)

    X_test = tfidf_vectorizer.transform([texte_pretraite])

    # Vérifier la cohérence des dimensions
    if X_test.shape[1] != model.n_features_in_:
        return {"error": f"Le modèle attend {model.n_features_in_} features, mais en a reçu {X_test.shape[1]}"}

    # Prédire les probabilités (au lieu de predict)
    proba = model.predict_proba(X_test)  # shape: (1, n_tags)

    # On récupère les indices des 5 tags les plus probables
    top_indices = proba[0].argsort()[::-1][:5]

    # Convertir indices en tags
    all_tags = mlb.classes_
    top_tags = [all_tags[i] for i in top_indices]

    return {"prediction": top_tags}