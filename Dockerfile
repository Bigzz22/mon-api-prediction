# Utilise l'image officielle Python
FROM python:3.10-slim

# Définir le dossier de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
COPY ui_streamlit.py .

# Installer les dépendances
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pytest

COPY . .

EXPOSE 8501

# Run tests before starting the app
RUN pytest tests/

CMD ["streamlit", "run", "ui_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
