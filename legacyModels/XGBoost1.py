import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib  # Für das Speichern und Laden der Modelle

from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from xgboost import Booster
from sklearn.model_selection import GridSearchCV

# Modellpfad
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

#################################### Trainingsdaten
# Historische Daten einlesen
pathHistorisch = '/Users/felixroeder/Documents/Stromverbrauch Vorhersage/Regionsdaten 17-22'  # Pfad zu historischen Dateien
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
weather_data_path = "/Users/felixroeder/Documents/Stromverbrauch Vorhersage/Wetterdaten.xlsx"
weather_df = pd.read_excel(weather_data_path, engine='openpyxl')
weather_df.columns = ['Station', 'Datum', 'Temperatur']
weather_df = weather_df.groupby('Datum')['Temperatur'].mean().reset_index()

# Feiertage einlesen
holiday_data_path = "/Users/felixroeder/Documents/Stromverbrauch Vorhersage/FeiertageDE.xlsx"
holiday_df = pd.read_excel(holiday_data_path, engine='openpyxl')
holiday_df.columns = ['Datum']

# Integration der Wetterdaten in den Hauptdatensatz
df_all = df_all.merge(weather_df, on='Datum', how='left')

# Konvertiere Datumsspalte zu Datetime und setze als Index
df_all['Datum'] = pd.to_datetime(df_all['Datum'], errors='coerce')
df_all.set_index('Datum', inplace=True)

# Zeilen mit leeren Werten entfernen
df_all = df_all.dropna()

######## Regionen vorerst zusammenführen
columns_to_average = [1, 2, 3, 5]

# Sichere die ursprünglichen Spaltennamen, da du mit numerischen Indizes arbeitest
selected_columns = df_all.columns[columns_to_average]

# Gruppieren nach 'Region' und Durchschnittswerte berechnen
df_all = df_all.groupby('Datum')[selected_columns].mean()

# Feiertagsspalte hinzufügen
df_all['Ist_Feiertag'] = df_all.index.isin(holiday_df['Datum']).astype(int)

# Zeitabhängige Features hinzufügen
df_all['Tag'] = df_all.index.day
df_all['Monat'] = df_all.index.month
df_all['Jahr'] = df_all.index.year
df_all['Ist_Wochenende'] = (df_all.index.weekday >= 5).astype(int)
df_all['Wochentag'] = df_all.index.weekday

saved_df = df_all.copy()

##################################### Testdaten 2023
data_path_test = '/Users/felixroeder/Documents/Stromverbrauch Vorhersage/Regionsdaten 23'

files_test = [f for f in os.listdir(data_path_test) if f.endswith('.xlsx')]

regions_test = []
data_by_region_test = {}

# Daten pro Region laden
for file in files_test:
    region_name = os.path.splitext(file)[0]  # Region aus Dateinamen ableiten
    regions_test.append(region_name)
    file_path = os.path.join(data_path_test, file)
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        df['Region'] = region_name
        data_by_region_test[region_name] = df
    except Exception as e:
        print(f"Fehler beim Laden der Datei {file}: {e}")

# Regionale Daten kombinieren
df_all_test = pd.concat(data_by_region_test.values(), ignore_index=True)

# Wetterdaten einlesen und durchschnitt bilden, Spalte Stationen entfernen
weather_data_path_test = "/Users/felixroeder/Documents/Stromverbrauch Vorhersage/Wetterdaten23.xlsx"
weather_df_test = pd.read_excel(weather_data_path_test, engine='openpyxl')
weather_df_test.columns = ['Station', 'Datum', 'Temperatur']
weather_df_test = weather_df_test.groupby('Datum')['Temperatur'].mean().reset_index()

# Feiertage einlesen
holiday_data_path_test = "/Users/felixroeder/Documents/Stromverbrauch Vorhersage/FeiertageDE23.xlsx"
holiday_df_test = pd.read_excel(holiday_data_path_test, engine='openpyxl')
holiday_df_test.columns = ['Datum']

# Integration der Wetterdaten in den Hauptdatensatz
df_all_test = df_all_test.merge(weather_df_test, on='Datum', how='left')

# Konvertiere Datumsspalte zu Datetime und setze als Index
df_all_test['Datum'] = pd.to_datetime(df_all_test['Datum'], errors='coerce')
df_all_test.set_index('Datum', inplace=True)

# Zeilen mit leeren Werten entfernen
df_all_test = df_all_test.dropna()

######## Regionen vorerst zusammenführen
columns_to_average = [1, 2, 3, 5]

# Sichere die ursprünglichen Spaltennamen, da du mit numerischen Indizes arbeitest
selected_columns = df_all_test.columns[columns_to_average]

# Gruppieren nach 'Region' und Durchschnittswerte berechnen
df_all_test = df_all_test.groupby('Datum')[selected_columns].mean()

# Feiertagsspalte hinzufügen
df_all_test['Ist_Feiertag'] = df_all_test.index.isin(holiday_df_test['Datum']).astype(int)

# Zeitabhängige Features hinzufügen
df_all_test['Tag'] = df_all_test.index.day
df_all_test['Monat'] = df_all_test.index.month
df_all_test['Jahr'] = df_all_test.index.year
df_all_test['Ist_Wochenende'] = (df_all_test.index.weekday >= 5).astype(int)
df_all_test['Wochentag'] = df_all_test.index.weekday


################ MODELL #############################

# Zielvariablen und Features
variables = ['Gesamt (Netzlast) [MWh]', 'Residuallast [MWh]', 'Pumpspeicher [MWh]']
features = ['Tag', 'Monat', 'Jahr', 'Ist_Wochenende', 'Temperatur', 'Ist_Feiertag', 'Wochentag']

# Zielvariable auswählen
selected_variable = st.selectbox("Wähle eine Zielvariable:", variables)

# Separate Skalierer für jede Zielvariable
scalers = {var: MinMaxScaler(feature_range=(0, 1)) for var in variables}

# Zielvariablen einzeln skalieren
for var in variables:
    df_all[[var]] = scalers[var].fit_transform(df_all[[var]])

# Skalieren der Features
scaler_features = MinMaxScaler(feature_range=(0, 1))
df_all[features] = scaler_features.fit_transform(df_all[features])

X_train = df_all[features]
y_train = df_all[variables]

# Modelle für jede Zielvariable trainieren
models = {}
for i, variable in enumerate(variables):
    model_path = os.path.join(model_dir, f"xgboost_{variable}.json")
    
    if os.path.exists(model_path):
        # Lade den Booster direkt
        booster = Booster()
        booster.load_model(model_path)
        
        # Verwende den Booster im XGBRegressor
        model = XGBRegressor(objective='reg:squarederror', n_estimators=50)
        model._Booster = booster
    else:
        model = XGBRegressor(objective='reg:squarederror', n_estimators=50, learning_rate=0.1, max_depth=5)
        model.fit(X_train, y_train[variable])
        model.save_model(model_path)
        
    models[variable] = model

# Vorhersage für 2023 (Testdaten)
X_2023 = df_all_test[features]

predictions_2023 = {}
for variable in variables:
    predictions_2023[variable] = models[variable].predict(X_2023)

    # Rückskalierung der Vorhersagen
    predictions_2023[variable] = scalers[variable].inverse_transform(predictions_2023[variable].reshape(-1, 1)).flatten()

# Erstelle DataFrame für Vorhersagen
predictions_df_2023 = pd.DataFrame(index=df_all_test.index)
for variable in variables:
    predictions_df_2023[f"Vorhersage_{variable}"] = predictions_2023[variable]

############################################################################# Visualisierungen

############## Trainingsdaten 2017-2022
train_data = saved_df[[selected_variable]]
train_data = train_data.rename(columns={selected_variable: "Tatsächlich"})

# Datumsbereich-Widget für die Trainingsdaten
start_date_train, end_date_train = st.date_input(
    "Wähle den Betrachtungszeitraum für die Trainingsdaten:",
    value=(df_all.index.min(), df_all.index.max()),
    min_value=df_all.index.min(),
    max_value=df_all.index.max(),
)

# Umwandlung der ausgewählten Daten in das datetime64-Format
start_date_train = pd.to_datetime(start_date_train)
end_date_train = pd.to_datetime(end_date_train)

# Aggregation der Trainingsdaten auf Tagesebene
train_data_daily = train_data.resample('D').mean()

# Filtere die Trainingsdaten basierend auf dem gewählten Zeitraum
filtered_train_data = train_data_daily[(train_data_daily.index >= start_date_train) & (train_data_daily.index <= end_date_train)]

# Visualisierung der gefilterten Trainingsdaten als Liniendiagramm
st.write(f"Gefilterte Trainingsdaten ({start_date_train.date()} bis {end_date_train.date()}) für {selected_variable}")
st.line_chart(filtered_train_data[['Tatsächlich']])

############## Testdaten Vergleich mit Vorhersage (2023)
# Diagramm für Testdaten (2023) mit Vorhersagen
test_data = df_all_test[[selected_variable]]
test_data['Vorhersage'] = predictions_2023[selected_variable]
test_data = test_data.rename(columns={selected_variable: "Tatsächlich"})

# Datumsbereich-Widget für die Testdaten
start_date_test, end_date_test = st.date_input(
    "Wähle den Betrachtungszeitraum für die Testdaten:",
    value=(df_all_test.index.min(), df_all_test.index.max()),
    min_value=df_all_test.index.min(),
    max_value=df_all_test.index.max(),
)

# Umwandlung der ausgewählten Daten in das datetime64-Format für Testdaten
start_date_test = pd.to_datetime(start_date_test)
end_date_test = pd.to_datetime(end_date_test)

# Aggregation der Testdaten auf Tagesebene
test_data_daily = test_data.resample('D').mean()

# Filtere die Testdaten basierend auf dem gewählten Zeitraum
filtered_test_data = test_data_daily[(test_data_daily.index >= start_date_test) & (test_data_daily.index <= end_date_test)]

# Visualisierung der gefilterten Testdaten als Liniendiagramm
st.write(f"Gefilterte Testdaten ({start_date_test.date()} bis {end_date_test.date()}) für {selected_variable} mit Vorhersagen")
st.line_chart(filtered_test_data[['Tatsächlich', 'Vorhersage']])

# Anzeige der Vorhersagen für Testdaten 2023
st.write("Vorhersagen für Testdaten 2023:", predictions_df_2023)

# Berechnung der Performance-Metriken (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Berechnung des MAPE für jedes Modell und Ausgabe in Streamlit
mape_values_2023 = {}
for variable in variables:
    mape = mean_absolute_percentage_error(df_all_test[variable], predictions_2023[variable])
    mape_values_2023[variable] = mape

st.write("MAPE pro Zielvariable für Testdaten 2023:", mape_values_2023)

# Erstelle DataFrame mit tatsächlichen und vorhergesagten Werten
results_2023 = pd.DataFrame(index=df_all_test.index)

for variable in variables:
    results_2023[f"Tatsächlich_{variable}"] = df_all_test[variable]
    results_2023[f"Vorhersage_{variable}"] = predictions_2023[variable]

# Ausgabe des DataFrames in Streamlit
st.write("Vergleich der tatsächlichen und vorhergesagten Werte für Testdaten 2023:", results_2023)
