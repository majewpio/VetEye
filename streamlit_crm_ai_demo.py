
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Vet-Eye AI CRM", layout="wide")

# --- LOGOWANIE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Vet-Eye CRM – Logowanie")
    username = st.text_input("Login", value="handlowiec")
    password = st.text_input("Hasło", type="password")
    if st.button("Zaloguj się"):
        if username == "handlowiec" and password == "vet123":
            st.session_state.logged_in = True
        else:
            st.error("Błędny login lub hasło.")

if not st.session_state.logged_in:
    login()
    st.stop()

# --- PULPIT CRM ---
st.title("📊 Vet-Eye CRM – Panel użytkownika")
st.sidebar.success("Zalogowano jako: handlowiec")

uploaded_file = st.sidebar.file_uploader("📥 Wgraj plik CSV z leadami", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Lista leadów handlowych")
    st.dataframe(df[['Nazwa kliniki', 'Województwo', 'Typ kliniki', 'Źródło leada']])

    if st.button("🚀 Wykonaj scoring AI"):
        with st.spinner("Model AI trenuje się i analizuje dane..."):

            # Przygotowanie danych
            df_encoded = pd.get_dummies(df, columns=['Województwo', 'Typ kliniki', 'Źródło leada'])
            X = df_encoded.drop(columns=['Nazwa kliniki', 'Kupiono'], errors='ignore')
            y = df_encoded['Kupiono'] if 'Kupiono' in df_encoded.columns else None

            # Trening modelu
            if y is not None:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                model.fit(X_train, y_train)
                acc = accuracy_score(y_test, model.predict(X_test))
                st.success(f"✅ Model wytrenowany – dokładność: {acc:.2%}")
            else:
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                model.fit(X, np.random.randint(0, 2, size=len(X)))

            # Predykcja
            scoring = model.predict_proba(X)[:, 1]
            df['Scoring AI (0-100)'] = (scoring * 100).round(1)
            df_sorted = df.sort_values(by='Scoring AI (0-100)', ascending=False)

            # Wizualna tabela
            st.subheader("🔍 Ranking leadów według AI")
            def kolor(val):
                if val > 75: return 'background-color: #b6fcb6'
                elif val > 50: return 'background-color: #fffbbb'
                else: return 'background-color: #fcb6b6'
            st.dataframe(df_sorted[['Nazwa kliniki', 'Scoring AI (0-100)', 'Demo przeprowadzone', 'Liczba rozmów', 'Liczba maili']].style.applymap(kolor, subset=['Scoring AI (0-100)']))

            # Wykres ważności cech
            st.subheader("📈 Najważniejsze cechy wpływające na scoring")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            features = X.columns[indices]
            plt.figure(figsize=(10, 5))
            sns.barplot(x=importances[indices], y=features)
            st.pyplot(plt)
else:
    st.info("Wgraj plik CSV z leadami, aby rozpocząć analizę.")
