import streamlit as st
import requests

st.title("Prédicteur de tags par IA")

st.markdown("Remplis les champs ci-dessous pour obtenir des prédictions de tags basées sur le titre et le contenu.")

# Champs d'entrée
titre = st.text_input("Titre")
texte = st.text_area("Texte")

# Bouton de prédiction
if st.button("Lancer la prédiction"):
    if not titre or not texte:
        st.warning("Merci de renseigner à la fois le titre et le texte.")
    else:
        # Préparer les données
        input_data = {
            "titre": titre,
            "texte": texte
        }

        # Envoyer la requête à ton API (en local ou sur Azure)
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                #"https://mon-api-prediction-hjcybjc4gad9cngv.westeurope-01.azurewebsites.net/predict",
                json=input_data
            )

            if response.status_code == 200:
                prediction = response.json().get("prediction", [])
                st.success("Prédictions trouvées :")
                for tag in prediction:
                    st.markdown(f"**{tag}**")
            else:
                st.error(f"Erreur {response.status_code} : {response.text}")
        except Exception as e:
            st.error(f"Erreur lors de la requête : {e}")