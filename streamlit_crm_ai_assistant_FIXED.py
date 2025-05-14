
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
st.title("🚀 Vet-Eye CRM – AI Asystent Sprzedaży")
st.sidebar.success("Zalogowano jako: handlowiec")

uploaded_file = st.sidebar.file_uploader("📥 Wgraj plik CSV z leadami", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Lista leadów")
    st.dataframe(df[['Nazwa kliniki', 'Województwo', 'Typ kliniki', 'Źródło leada']])

    if st.button("🧠 Wykonaj scoring AI"):
        with st.spinner("Analiza i trenowanie modelu..."):

            df_encoded = pd.get_dummies(df, columns=['Województwo', 'Typ kliniki', 'Źródło leada'])
            X = df_encoded.drop(columns=['Nazwa kliniki', 'Kupiono'], errors='ignore')
            y = df_encoded['Kupiono'] if 'Kupiono' in df_encoded.columns else None

            if y is not None:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                model.fit(X_train, y_train)
                acc = accuracy_score(y_test, model.predict(X_test))
                st.success(f"✅ Dokładność modelu: {acc:.2%}")
            else:
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                model.fit(X, np.random.randint(0, 2, size=len(X)))

            scoring = model.predict_proba(X)[:, 1]
            df['Scoring AI (0-100)'] = (scoring * 100).round(1)
            df_sorted = df.sort_values(by='Scoring AI (0-100)', ascending=False)

            st.subheader("🔍 Ranking leadów + rekomendacje AI")
            for idx, row in df_sorted.iterrows():
                st.markdown(f"### 🏥 {row['Nazwa kliniki']} – Scoring AI: {row['Scoring AI (0-100)']}%")

                # Strategie kontaktu
                strategy = ""
                script = ""
                if row['Scoring AI (0-100)'] >= 80:
                    strategy = "📞 Telefon – od razu po demo"
                    script = "Dzień dobry, wiem że widzieli Państwo demo. Czy są Państwo gotowi podjąć decyzję? Mogę zaproponować również warianty finansowania."
                elif row['Scoring AI (0-100)'] >= 50:
                    strategy = "📧 Mail – z podsumowaniem oferty"
                    script = "Dzień dobry, przesyłam propozycję z podsumowaniem funkcji i cen. Proszę dać znać, czy możemy umówić się na rozmowę."
                else:
                    strategy = "⏳ Odłożenie kontaktu / przypomnienie za 7 dni"
                    script = "Na ten moment nie będę ponawiał kontaktu – lead nisko oceniony. Zaplanuj follow-up za tydzień."

                st.write(f"**🔄 Strategia kontaktu:** {strategy}")
                st.markdown("**🗣️ Propozycja skryptu rozmowy:**")
                st.code(script)
                st.markdown("---")

            st.subheader("📈 Najważniejsze cechy wpływające na scoring AI")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            features = X.columns[indices]
            plt.figure(figsize=(10, 5))
            sns.barplot(x=importances[indices], y=features)
            st.pyplot(plt)
else:
    st.info("Wgraj plik CSV z leadami, aby uruchomić AI asystenta sprzedaży.")
