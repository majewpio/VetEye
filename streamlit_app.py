
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Vet-Eye AI CRM", layout="wide")
st.title("🧠 Vet-Eye AI CRM – Lead Scoring z wykorzystaniem AI")

# Upload CSV
uploaded_file = st.file_uploader("📥 Wgraj plik CSV z leadami (np. leady_veteye_demo.csv)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dane wczytane poprawnie.")
    st.dataframe(df.head(10))

    if st.button("🚀 Wykonaj scoring AI"):
        with st.spinner("Model AI trenuje się i analizuje dane..."):

            # Przygotowanie danych
            df_encoded = pd.get_dummies(df, columns=['Województwo', 'Typ kliniki', 'Źródło leada'])
            X = df_encoded.drop(columns=['Nazwa kliniki', 'Kupiono'], errors='ignore')
            y = df_encoded['Kupiono'] if 'Kupiono' in df_encoded.columns else None

            # Jeśli mamy etykiety, dzielimy na train/test
            if y is not None:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.info(f"🎯 Dokładność modelu na danych testowych: {acc:.2%}")
            else:
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                model.fit(X, np.random.randint(0, 2, size=len(X)))  # sztuczne etykiety jeśli brak

            # Scoring
            scoring = model.predict_proba(X)[:, 1]
            df['Scoring AI (0-100)'] = (scoring * 100).round(1)
            df_sorted = df.sort_values(by='Scoring AI (0-100)', ascending=False)

            # Kolory
            def kolor_wiersza(val):
                if val > 75:
                    return 'background-color: #b6fcb6'
                elif val > 50:
                    return 'background-color: #fffbbb'
                else:
                    return 'background-color: #fcb6b6'

            st.subheader("📊 Wyniki scoringu AI")
            st.dataframe(df_sorted.style.applymap(kolor_wiersza, subset=['Scoring AI (0-100)']))

            # Feature Importance
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            features = X.columns[indices]
            plt.figure(figsize=(10, 5))
            sns.barplot(x=importances[indices], y=features)
            plt.title("🔍 Najważniejsze cechy wpływające na scoring")
            st.pyplot(plt)
