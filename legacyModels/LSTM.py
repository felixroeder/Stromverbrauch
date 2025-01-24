import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Modellpfad
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# historische Daten einlesen
pathHistorisch = '/Users/felixroeder/Documents/Stromverbrauch Vorhersage/Verbrauchsdaten nach Region'  # Pfad zu historischen Dateien
files = [f for f in os.listdir(pathHistorisch) if f.endswith('.xlsx')]

regions = []
data_by_region = {}

# Daten pro Region laden
for file in files:
    region_name = os.path.splitext(file)[0]  # Region aus Dateinamen ableiten
    regions.append(region_name)
    file_path = os.path.join(pathHistorisch, file)
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        df['Region'] = region_name
        data_by_region[region_name] = df
    except Exception as e:
        print(f"Fehler beim Laden der Datei {file}: {e}")

# Regionale Daten kombinieren
df_all = pd.concat(data_by_region.values(), ignore_index=True)

# Wetterdaten einlesen
weather_data_path = "/Users/felixroeder/Documents/Stromverbrauch Vorhersage/Wetterdaten.xlsx"  # Pfad zu Wetterdaten
weather_df = pd.read_excel(weather_data_path, engine='openpyxl')
weather_df.columns = ['Station', 'Datum', 'Temperatur']

# Feiertage einlesen
holiday_data_path = "/Users/felixroeder/Documents/Stromverbrauch Vorhersage/FeiertageDE.xlsx"
holiday_df = pd.read_excel(holiday_data_path, engine='openpyxl')
holiday_df.columns = ['Datum']

# Gruppierung nach Datum und Berechnung der Durchschnittstemperatur
average_temp_df = weather_df.groupby('Datum')['Temperatur'].mean().reset_index()
average_temp_df.columns = ['Datum', 'Temperatur']
average_temp_df = average_temp_df.sort_values(by='Datum')

# Integration der Wetterdaten in den Hauptdatensatz
df_all = df_all.merge(average_temp_df, on='Datum', how='left')

# Konvertiere Datumsspalte zu Datetime und setze als Index
df_all['Datum'] = pd.to_datetime(df_all['Datum'], errors='coerce')
df_all.set_index('Datum', inplace=True)

# Feiertagsspalte hinzufügen
df_all['Ist_Feiertag'] = df_all.index.isin(holiday_df['Datum']).astype(int)

# Zeitabhängige Features hinzufügen
df_all['Tag'] = df_all.index.day
df_all['Monat'] = df_all.index.month
df_all['Jahr'] = df_all.index.year
df_all['Ist_Wochenende'] = (df_all.index.weekday >= 5).astype(int)
df_all['Wochentag'] = (df_all.index.weekday)

# Zielvariablen und Regionen
variables = ['Gesamt (Netzlast) [MWh]', 'Residuallast [MWh]', 'Pumpspeicher [MWh]']
regions = list(data_by_region.keys())

# Skalierung der Daten
features = variables + ['Tag', 'Monat', 'Jahr', 'Ist_Wochenende', 'Temperatur', 'Ist_Feiertag', 'Wochentag']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_all[features])

st.write(df_all)
st.write(scaled_data)

# Trainings- und Testdaten aufbereiten
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Rolling Dataset erstellen
def create_rolling_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step])
        y.append(data[i + time_step, :len(variables)])  # Nur Zielvariablen als y
    return np.array(X), np.array(y)

# Zeitfenster für Rolling Dataset festlegen
time_step = 30
X_train, y_train = create_rolling_dataset(train_data, time_step)
X_test, y_test = create_rolling_dataset(test_data, time_step)

# Modell laden oder trainieren
model_path = os.path.join(model_dir, f"Model.h5")
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(32),
        Dense(len(variables))
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Modelltraining mit Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

    # Modell speichern
    model.save(model_path)

# Vorhersagen für 2023
predicted_data_scaled = model.predict(X_test[-365:])
predicted_data = scaler.inverse_transform(
    np.hstack([predicted_data_scaled, np.zeros((365, scaled_data.shape[1] - len(variables)))]))

####################################################################################################################
# Datensatz für performance Messung
performanceData2023 = "/Users/felixroeder/Documents/Stromverbrauch Vorhersage/Testdaten2023.xlsx"  # Pfad zu Wetterdaten
performance_df = pd.read_excel(performanceData2023, engine='openpyxl')
performance_df.columns = ['Datum', 'Wochentag', 'Gesamt (Netzlast) [MWh]', 'Residuallast [MWh]', 'Pumpspeicher [MWh]']

# MAPE (Mean Absolute Percentage Error) berechnen
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Extrahieren der tatsächlichen Werte für alle Zielvariablen
actual_values = performance_df[['Gesamt (Netzlast) [MWh]', 'Residuallast [MWh]', 'Pumpspeicher [MWh]']].values

# Extrahieren der vorhergesagten Werte für alle Zielvariablen
predicted_values = predicted_data[:, :3]  # Vorhersagen für die ersten drei Variablen

# MAPE für jede Zielvariable berechnen
mape_values = []
for i, variable in enumerate(['Gesamt (Netzlast) [MWh]', 'Residuallast [MWh]', 'Pumpspeicher [MWh]']):
    mape = mean_absolute_percentage_error(actual_values[:, i], predicted_values[:, i])
    mape_values.append(mape)

# Durchschnittlichen MAPE für alle Zielvariablen berechnen
average_mape = np.mean(mape_values)
st.write(average_mape)


####################################################################################################################
# Streamlit: Visualisierung
st.title("Stromverbrauchsvorhersage 2023")

# Auswahlmöglichkeiten für Zielvariablen
selected_variables = st.multiselect("Wähle Zielvariablen aus:", variables, default=variables[:1])

# Zeitraum-Einschränkung
start_date = st.date_input("Startdatum", value=pd.Timestamp('2023-01-01'), min_value=pd.Timestamp('2023-01-01'), max_value=pd.Timestamp('2023-12-31'))
end_date = st.date_input("Enddatum", value=pd.Timestamp('2023-12-31'), min_value=pd.Timestamp('2023-01-01'), max_value=pd.Timestamp('2023-12-31'))

# Erstellung der Zeitreihe für 2023
dates_2023 = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

# Filtern der Daten basierend auf dem gewählten Zeitraum
mask = (dates_2023 >= pd.Timestamp(start_date)) & (dates_2023 <= pd.Timestamp(end_date))
filtered_dates = dates_2023[mask]

# Berechnungen für den Stromverbrauch
all_consumption_data = []

for variable in selected_variables:
    variable_index = variables.index(variable)
    values = predicted_data[mask, variable_index]
    
    # Daten für Diagramm und Analyse sammeln
    all_consumption_data.extend(zip(filtered_dates, values, [variable] * len(values)))
    
    # Konsistente Berechnungen für Statistiken
    max_value = np.max(values)
    min_value = np.min(values)
    avg_value = np.mean(values)
    total_value = np.sum(values)
    
    # Anzeigen der berechneten Werte
    st.subheader(f"Stromverbrauchsstatistiken für {variable}")
    st.write(f"Durchschnittlicher Verbrauch: {avg_value:.2f} MWh")
    st.write(f"Maximaler Verbrauch: {max_value:.2f} MWh")
    st.write(f"Minimaler Verbrauch: {min_value:.2f} MWh")
    st.write(f"Gesamter Verbrauch im Zeitraum: {total_value:.2f} MWh")

# DataFrame zur Analyse und Anzeige der Top 5 Tage
consumption_df = pd.DataFrame(all_consumption_data, columns=["Datum", "Verbrauch", "Variable"])

# Diagramm erstellen
fig, ax = plt.subplots(figsize=(12, 6))
for variable in selected_variables:
    df_variable = consumption_df[consumption_df["Variable"] == variable]
    
    ax.plot(df_variable["Datum"], df_variable["Verbrauch"], label=f"{variable} (Verbrauch)")
    
    ax.axhline(df_variable["Verbrauch"].max(), color='red', linestyle='--', label=f"{variable} Max")
    ax.axhline(df_variable["Verbrauch"].min(), color='green', linestyle='--', label=f"{variable} Min")
    ax.axhline(df_variable["Verbrauch"].mean(), color='blue', linestyle='--', label=f"{variable} Durchschnitt")

# Achsentitel und Labels
ax.set_title("Stromverbrauch 2023 - Maximal, Minimal und Durchschnitt")
ax.set_xlabel("Datum")
ax.set_ylabel("Verbrauch (MWh)")
ax.set_xticks(filtered_dates[::10])
ax.set_xticklabels([d.strftime("%d.%m.%Y") for d in filtered_dates[::10]], rotation=45, ha='right')
ax.legend()
st.pyplot(fig)

# Top 5 Tage mit maximalem und minimalem Verbrauch pro Variable anzeigen
st.subheader("Top 5 Tage mit dem höchsten und niedrigsten Verbrauch")

for variable in selected_variables:
    st.write(f"Top 5 Tage mit dem höchsten Verbrauch für {variable}:")
    st.write(consumption_df[consumption_df["Variable"] == variable]
             .sort_values(by='Verbrauch', ascending=False)
             .head(5))
    
    st.write(f"Top 5 Tage mit dem niedrigsten Verbrauch für {variable}:")
    st.write(consumption_df[consumption_df["Variable"] == variable]
             .sort_values(by='Verbrauch', ascending=True)
             .head(5))