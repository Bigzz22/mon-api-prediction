# 🔖 Suggestion Automatique de Tags Stack Overflow

Ce projet vise à développer un moteur d'inférence capable de suggérer automatiquement les tags les plus pertinents pour une question Stack Overflow (titre + corps de la question). Il s'agit d'une **classification multi-label**, encapsulée dans une **API FastAPI**, déployée via **Docker** sur **Azure App Service**, et connectée à une **interface utilisateur Streamlit**.

---

## 🚀 Objectifs du projet

- Collecter des questions Stack Overflow via leur API
- Nettoyer et prétraiter les textes (Bag-of-Words, CountVectorizer)
- Entraîner un modèle de classification multi-label (Logistic Regression + OneVsRest)
- Déployer le modèle dans une API FastAPI
- Offrir une interface utilisateur avec Streamlit
- Mettre en place une stratégie de **MLOps** (suivi de dérive, tests, versioning)
- Suivre les performances dans le temps avec **MLflow Tracking**

---

## 🗂️ Structure du projet

```
.
├── app.py                     # API FastAPI exposant l'endpoint /predict
├── ui_streamlit.py            # Interface utilisateur Streamlit
├── model/
│   ├── model.pkl              # Modèle ML sérialisé
│   └── vectorizer.pkl         # Vectoriseur (CountVectorizer / TF-IDF)
├── data/
│   ├── questions.csv          # Données de Stack Overflow
│   └── monthly_scores/        # Scores de stabilité mensuels
├── tests/
│   └── test_main.py           # Tests unitaires pour l’API
├── notebooks/
│   └── exploration.ipynb      # Analyse exploratoire et prétraitement
├── Dockerfile                 # Fichier Docker pour le déploiement
├── requirements.txt           # Liste des dépendances Python
└── README.md                  # Ce fichier
```

---

## ⚙️ Installation locale

```bash
git clone https://github.com/ton-compte/mon-api-prediction.git
cd mon-api-prediction
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

---

## 🧪 Lancer les tests unitaires

```bash
pytest tests/
```

---

## 🌐 Lancer l'API en local

```bash
uvicorn app:app --reload
```

---

## 🎛️ Interface utilisateur Streamlit

```bash
streamlit run ui_streamlit.py
```

---

## 📦 Déploiement

L’API est containerisée avec **Docker**, puis déployée sur **Azure App Service**.

```bash
docker build -t bigzz22/streamlit-ui .
docker push bigzz22/streamlit-ui
```

---

## 📉 Suivi des performances

Le projet intègre :

- 📊 **MLflow Tracking** pour suivre les scores (Jaccard, F1, etc.) mensuels
- 📈 Suivi des dérives via **Evidently**, **Popmon** et/ou **Prometheus + Grafana**

---

## 📚 Exemple d'appel à l'API

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"titre": "How to filter a pandas DataFrame?", "texte": "I want to filter rows based on column values."}'
```

Réponse attendue :

```json
{
  "prediction": ["pandas", "filter", "dataframe", "python", "data-cleaning"]
}
```

---

## 📦 Dépendances clés

Voir [`requirements.txt`](./requirements.txt)

---

## 🛡️ RGPD et Éthique

- Seuls les **données textuelles** des questions Stack Overflow sont utilisées.
- Aucune information personnelle, ID utilisateur, IP ou métadonnée identifiante n'est collectée.

---

## ✍️ Auteurs

- Bigot Benjamin  
- Projet dans le cadre du parcours Data Scientist — OpenClassrooms

---

## 📃 Licence

[MIT License](LICENSE)
