
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Ustawienia strony ---
st.set_page_config(page_title="Vet-Eye CRM AI", layout="wide")

# --- Styl i estetyka ---
st.markdown("""
    <style>
    .big-font {{
        font-size: 25px !important;
        font-weight: bold;
    }}
    .center {{
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    </style>
""", unsafe_allow_html=True)

# --- LOGOWANIE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.image("piesek_logowanie.JPG", width=500)
    st.markdown('<p class="big-font">Vet-Eye CRM – Zaloguj się</p>', unsafe_allow_html=True)
    username = st.text_input("Login", placeholder="np. handlowiec")
    password = st.text_input("Hasło", type="password")
    if st.button("Zaloguj się"):
        if username == "handlowiec" and password == "vet123":
            st.session_state.logged_in = True
        else:
            st.error("Nieprawidłowy login lub hasło.")

if not st.session_state.logged_in:
    login()
    st.stop()

# --- Wczytanie leadów z pliku CSV (bez ładowania przez użytkownika) ---
df = pd.read_csv("leady_veteye_demo.csv")

st.title("📊 Vet-Eye AI CRM – Asystent Sprzedaży")
st.write("Poniżej prezentujemy 5 najlepszych leadów z predefiniowaną strategią kontaktu i rekomendacją produktu.")

# --- Model AI ---
df_encoded = pd.get_dummies(df, columns=['Województwo', 'Typ kliniki', 'Źródło leada'])
X = df_encoded.drop(columns=['Nazwa kliniki', 'Kupiono'], errors='ignore')
y = df_encoded['Kupiono'] if 'Kupiono' in df_encoded.columns else np.random.randint(0, 2, size=len(X))

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)
scoring = model.predict_proba(X)[:, 1]
df['Scoring AI (0-100)'] = (scoring * 100).round(1)
df_sorted = df.sort_values(by='Scoring AI (0-100)', ascending=False).head(5)

# --- Prezentacja leadów + rekomendacje ---
for idx, row in df_sorted.iterrows():
    st.markdown(f"## 🏥 {row['Nazwa kliniki']} – Scoring AI: {row['Scoring AI (0-100)']}%")

    if row['Scoring AI (0-100)'] >= 80:
        strategy = "📞 Telefon – klient prawie zdecydowany"
        script = "Proszę zapytać o finalną decyzję. Zaproponuj wariant płatności lub rabat."
        product = "Vet Pro 70"
        product_img = "produkt_2.JPG"
        product_desc = "Zaawansowane urządzenie USG dla dużych klinik – funkcje Dopplera, 3 sondy, ekran dotykowy."
    elif row['Scoring AI (0-100)'] >= 50:
        strategy = "📧 E-mail z ofertą i linkiem do demo"
        script = "Zachęć klienta do ponownego obejrzenia materiałów. Zaproponuj demo online."
        product = "Vet Portable 15"
        product_img = "produkt_1.JPG"
        product_desc = "Lekki, przenośny aparat idealny do wizyt terenowych i mniejszych klinik."
    else:
        strategy = "⏳ Odłożenie kontaktu i przypomnienie za 10 dni"
        script = "Warto poczekać – klient może nie być jeszcze gotowy."
        product = "Brak rekomendacji produktu na tym etapie"
        product_img = None
        product_desc = ""

    st.markdown(f"**🔄 Strategia kontaktu:** {strategy}")
    st.markdown("**🗣️ Skrypt rozmowy:**")
    st.code(script)
    st.markdown(f"**🖥️ Rekomendowany produkt:** {product}")
    if product_img:
        st.image(product_img, width=400, caption=product_desc)
    st.markdown("---")

# --- Wykres wpływu cech ---
st.subheader("📈 Najważniejsze cechy wpływające na scoring AI")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
features = X.columns[indices]
plt.figure(figsize=(10, 5))
sns.barplot(x=importances[indices], y=features)
st.pyplot(plt)
