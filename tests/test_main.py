import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from app import app, preprocess_text, predict, InputData

client = TestClient(app)

def test_preprocess_output_is_string():
    assert isinstance(preprocess_text("Bonjour !"), str)

def test_preprocess_removes_punctuation():
    assert "?" not in preprocess_text("C'est quoi ?")

def test_preprocess_lowercase():
    assert preprocess_text("BONJOUR") == "bonjour"

def test_predict_tags():
    test_input = {
        "titre": "How to use pandas to filter data?",
        "texte": "I have a dataframe with many columns and want to filter rows where values are greater than 10."
    }

    response = client.post("/predict", json=test_input)

    # Vérifie que la requête s’est bien passée
    assert response.status_code == 200

    # Vérifie que la réponse est bien un JSON avec les bons champs
    result = response.json()
    assert isinstance(result, dict)
    assert "prediction" in result
    assert isinstance(result["prediction"], list)

    # Exemple : tu peux vérifier que certains tags attendus peuvent apparaître
    expected_tags = {"python", "pandas", "dataframe"}
    assert any(tag in result["prediction"] for tag in expected_tags)

def test_prediction_format():
    data = InputData(titre="How to sort a Python array ?", texte="I would like to sort a Python array.")
    result = predict(data)
    assert "prediction" in result
    assert isinstance(result["prediction"], list)

def test_prediction_length_limit():
    data = InputData(titre="test", texte="test")
    result = predict(data)
    assert len(result["prediction"]) <= 5


def test_api_response():
    response = client.post("/predict", json={
        "titre": "How to use pandas ?",
        "texte": "I don't understand how to use a Pandas dataframe."
    })
    assert response.status_code == 200
    assert "prediction" in response.json()