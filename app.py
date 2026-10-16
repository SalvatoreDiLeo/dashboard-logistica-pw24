import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# 1. IMPOSTAZIONI PAGINA
st.set_page_config(page_title="Gestione Scorte Cofra", layout="wide")
st.title("📦 Dashboard Previsione Scorte - PW24")
st.write("Progetto sviluppato per analizzare i volumi di magazzino e prevedere le scorte necessarie.")

# 2. CREAZIONE DEI DATI FINTI (Simulazione estrazione dal WMS)
@st.cache_data
def crea_dati_mensili():
    np.random.seed(42) # Fissa i numeri casuali per non farli cambiare a ogni refresh
    mesi = pd.date_range(start="2024-01-01", periods=24, freq="MS")

    # Genero i numeri base per i 4 articoli + un po' di "rumore" casuale per renderli reali
    monti = 1200 + np.random.normal(0, 50, 24) 
    isometry = np.linspace(500, 1500, 24) + np.random.normal(0, 30, 24) 
    parka = 800 + (400 * np.sin(np.pi * np.arange(24) / 6)) + np.random.normal(0, 40, 24) 
    welder = np.random.choice([0, 50, 200, 800], size=24, p=[0.5, 0.3, 0.1, 0.1])

    # Inserisco tutto in una tabella (DataFrame) e arrotondo i numeri
    tabella = pd.DataFrame({
        "Mese": mesi,
        "Monti S3 (Stazionario)": np.round(np.maximum(0, monti)),
        "Isometry ESD (Trend)": np.round(np.maximum(0, isometry)),
        "Parka Telsen (Stagionale)": np.round(np.maximum(0, parka)),
        "Welder Bis (Irregolare)": np.round(np.maximum(0, welder))
    })
    return tabella

# Salvo la tabella nella variabile 'dati'
dati = crea_dati_mensili()

# 3. MENU LATERALE (SIDEBAR)
st.sidebar.header("Parametri Magazzino")
articolo = st.sidebar.selectbox("Scegli l'articolo da analizzare:", dati.columns[1:])
lead_time_settimane = st.sidebar.slider("Lead Time Fornitore (Settimane):", 1, 12, 4)
livello_servizio = st.sidebar.slider("Livello di Servizio (%):", 80, 99, 95)

# Trasformo il livello in percentuale in un fattore statistico (Z-score)
z_score = norm.ppf(livello_servizio / 100.0)
mesi_lead_time = lead_time_settimane / 4.0

# 4. PREPARO I CALCOLI MATEMATICI
valori = dati[articolo].values  # Prendo solo i numeri della colonna scelta
n = len(valori)

# A) Calcolo Media Mobile
previsione_sma = np.zeros(n)
previsione_sma[0] = valori[0]
for i in range(1, n):
    if i < 3:
        previsione_sma[i] = np.mean(valori[:i])
    else:
        previsione_sma[i] = np.mean(valori[i-3:i])

# B) Calcolo Smussamento Esponenziale
previsione_ses = np.zeros(n)
previsione_ses[0] = valori[0]
alpha = 0.3
for i in range(1, n):
    previsione_ses[i] = (alpha * valori[i-1]) + ((1 - alpha) * previsione_ses[i-1])

# C) Calcolo Trend Lineare
x = np.arange(n)
coefficiente, intercetta = np.polyfit(x, valori, 1)
previsione_trend = (coefficiente * x) + intercetta

# 5. FUNZIONE PER CALCOLARE GLI ERRORI DEI MODELLI
def calcola_errori(reali, previsti):
    errore = reali - previsti
    mae = np.mean(np.abs(errore))
    mse = np.mean(errore**2)
    # Metto un "1" al posto degli "0" per non far bloccare il programma nella divisione
    reali_sicuri = np.where(reali == 0, 1, reali)
    mape = np.mean(np.abs(errore / reali_sicuri)) * 100
    return round(mae, 2), round(mse, 2), round(mape, 2)

# Uso la funzione per i 3 modelli
mae_sma, mse_sma, mape_sma = calcola_errori(valori, previsione_sma)
mae_ses, mse_ses, mape_ses = calcola_errori(valori, previsione_ses)
mae_trend, mse_trend, mape_trend = calcola_errori(valori, previsione_trend)

# Metto gli errori in una tabella
tabella_errori = pd.DataFrame({
    "Modello": ["Media Mobile", "Esponenziale", "Trend"],
    "MAE": [mae_sma, mae_ses, mae_trend],
    "MSE": [mse_sma, mse_ses, mse_trend],
    "MAPE (%)": [mape_sma, mape_ses, mape_trend]
})

# Trovo chi ha l'errore percentuale (MAPE) più basso
indice_migliore = tabella_errori["MAPE (%)"].idxmin()
modello_migliore = tabella_errori.loc[indice_migliore, "Modello"]

# Prendo l'ultima previsione utile dal modello vincitore
if indice_migliore == 0:
    stima_futura = previsione_sma[-1]
elif indice_migliore == 1:
    stima_futura = previsione_ses[-1]
else:
    stima_futura = previsione_trend[-1]

# 6. CALCOLO FINALE DELLE SCORTE FISICHE
deviazione_standard = np.std(valori)
scorta_ciclica = stima_futura / 2
scorta_sicurezza = z_score * deviazione_standard * np.sqrt(mesi_lead_time)
scorta_pipeline = stima_futura * mesi_lead_time

if "Stagionale" in articolo:
    scorta_anticipata = max(0, np.max(valori) - stima_futura)
else:
    scorta_anticipata = 0

# 7. DISEGNO L'INTERFACCIA STREAMLIT (Grafico e Risultati)
st.subheader("Andamento Temporale e Modelli")
grafico = go.Figure()
grafico.add_trace(go.Scatter(x=dati["Mese"], y=valori, mode="lines+markers", name="Reale", line=dict(color="black")))
grafico.add_trace(go.Scatter(x=dati["Mese"], y=previsione_sma, name="Media Mobile", line=dict(dash="dash")))
grafico.add_trace(go.Scatter(x=dati["Mese"], y=previsione_ses, name="Esponenziale", line=dict(dash="dot")))
grafico.add_trace(go.Scatter(x=dati["Mese"], y=previsione_trend, name="Trend"))
st.plotly_chart(grafico, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.write("**Errori dei modelli**")
    st.dataframe(tabella_errori, hide_index=True)
    st.success(f"Il sistema consiglia l'uso di: **{modello_migliore}**")

with col2:
    st.write("**Giacenze di Magazzino Calcolate**")
    st.info(f"📦 Scorta Ciclica: {int(scorta_ciclica)} unità")
    st.warning(f"🛡️ Scorta Sicurezza: {int(scorta_sicurezza)} unità")
    st.error(f"🚚 Merce in Transito (Pipeline): {int(scorta_pipeline)} unità")
    st.success(f"❄️ Scorta Anticipata: {int(scorta_anticipata)} unità")