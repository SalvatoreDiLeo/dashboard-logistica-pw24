import numpy as np
import pandas as pd

# Questo file salva in un CSV i dati simulati che usa la dashboard.
# Uso gli stessi numeri di crea dati mensili, con i cursori ai valori iniziali.
variabilita = 1
trend = 0
stagionalita = 0

np.random.seed(42)  # stesso seed della dashboard, così escono gli stessi dati
mesi = pd.date_range(start="2024-01-01", periods=36, freq="MS")
t = np.arange(36)
extra = (trend * t) + (stagionalita * np.sin(np.pi * t / 6))

# Le quattro serie, nello stesso ordine della dashboard
monti = 1200 + extra + np.random.normal(0, 50 * variabilita, 36)
isometry = np.linspace(500, 1500, 36) + extra + np.random.normal(0, 30 * variabilita, 36)
parka = 800 + (400 * np.sin(np.pi * t / 6)) + extra + np.random.normal(0, 40 * variabilita, 36)
welder = np.random.choice([0, 50, 200, 800], size=36, p=[0.5, 0.3, 0.1, 0.1])

# Tolgo i valori negativi, arrotondo e metto tutto in una tabella
dati = pd.DataFrame({
    "Mese": mesi.strftime("%Y-%m-%d"),
    "Monti S3 (Stazionario)": np.round(np.maximum(0, monti)).astype(int),
    "Isometry ESD (Trend)": np.round(np.maximum(0, isometry)).astype(int),
    "Parka Telsen (Stagionale)": np.round(np.maximum(0, parka)).astype(int),
    "Welder Bis (Irregolare)": np.round(np.maximum(0, welder)).astype(int),
})

dati.to_csv("dati_domanda_pw24.csv", index=False)
print("File salvato: dati_domanda_pw24.csv")