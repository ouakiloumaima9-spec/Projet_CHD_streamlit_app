import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# 1. Charger le modèle sauvegardé
# -----------------------------
@st.cache_resource
def load_model():
    with open("Model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("Prédiction du Risque de Maladie Cardiaque")
st.write("Cette application utilise un modèle entraîné pour prédire le risque de maladie cardiaque à partir de données cliniques.")

# -----------------------------
# 2. Interface de saisie utilisateur
# -----------------------------

st.header("Données du patient")

sbp = st.number_input("Pression systolique (sbp)", min_value=80, max_value=250, value=130)
ldl = st.number_input("Cholestérol LDL (ldl)", min_value=50, max_value=400, value=150)
adiposity = st.number_input("Adiposité", min_value=5.0, max_value=60.0, value=25.0)
obesity = st.number_input("Obésité", min_value=5.0, max_value=80.0, value=30.0)
age = st.number_input("Âge", min_value=18, max_value=100, value=50)

famhist = st.selectbox("Antécédents familiaux (famhist)", ["Absent", "Present"])

# -----------------------------
# 3. Construction dynamique d’un exemple
# -----------------------------

input_data = pd.DataFrame({
    'sbp': [sbp],
    'ldl': [ldl],
    'adiposity': [adiposity],
    'famhist': [famhist],
    'obesity': [obesity],
    'age': [age]
})

# -----------------------------
# 4. Prédiction et probabilité
# -----------------------------

if st.button("Prédire"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("Résultat de la prédiction :")
    if prediction == 1:
        st.error(f"Risque élevé de maladie cardiaque — Probabilité : {proba:.2f}")
    else:
        st.success(f"Risque faible de maladie cardiaque — Probabilité : {proba:.2f}")
