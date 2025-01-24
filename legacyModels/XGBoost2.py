import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib  # Für das Speichern und Laden der Modelle

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from xgboost import Booster
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error

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
df_all = df_all.merge(weather_df, on='Datum', how='left') # Integration in Hauptdatensatz

# Feiertage einlesen
holiday_data_path = "/Users/felixroeder/Documents/Stromverbrauch Vorhersage/FeiertageDE.xlsx"
holiday_df = pd.read_excel(holiday_data_path, engine='openpyxl')
holiday_df.columns = ['Datum']

# Bevölkerung einlesen
einwohner_data_path = "/Users/felixroeder/Documents/Stromverbrauch Vorhersage/Bevölkerung17-22.xlsx"
einwohner_df = pd.read_excel(einwohner_data_path, engine='openpyxl')
df_all = df_all.merge(einwohner_df, on='Datum', how='left') # Integration in Hauptdatensatz

# Reine Elekrokraftwagen Neuzulassungen in % einlesen (monatliche Basis)
elektroKraftwagen_data_path = "/Users/felixroeder/Documents/Stromverbrauch Vorhersage/NeuzulassungenKraftfahrt17-22.xlsx"
elektroKraftwagen_df = pd.read_excel(elektroKraftwagen_data_path, engine='openpyxl')
df_all = df_all.merge(elektroKraftwagen_df, on='Datum', how='left') # Integration in Hauptdatensatz

# Import und Export von Strom (monatliche Basis)
importexport_data_path = "/Users/felixroeder/Documents/Stromverbrauch Vorhersage/ExportImport17-22.xlsx"
importexport_df = pd.read_excel(importexport_data_path, engine='openpyxl')
df_all = df_all.merge(importexport_df, on='Datum', how='left') # Integration in Hauptdatensatz

# Konvertiere Datumsspalte zu Datetime und setze als Index
df_all['Datum'] = pd.to_datetime(df_all['Datum'], errors='coerce')
df_all.set_index('Datum', inplace=True)

# Zeilen mit leeren Werten entfernen
df_all = df_all.dropna()

######## Regionen vorerst zusammenführen
columns_to_average = [1, 2, 3, 5, 6, 7, 8, 9]

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

# Konvertiere Datumsspalte zu Datetime und setze als Index
df_all_test['Datum'] = pd.to_datetime(df_all_test['Datum'], errors='coerce')
df_all_test.set_index('Datum', inplace=True)

# Zeilen mit leeren Werten entfernen
df_all_test = df_all_test.dropna()

######## Regionen vorerst zusammenführen
columns_to_average = [1, 2, 3]

# Sichere die ursprünglichen Spaltennamen, da du mit numerischen Indizes arbeitest
selected_columns = df_all_test.columns[columns_to_average]

# Gruppieren nach 'Region' und Durchschnittswerte berechnen
df_all_test = df_all_test.groupby('Datum')[selected_columns].mean()

saved_df_test = df_all_test.copy()

################ MODELL #############################
# Zielvariablen und Features
variables = ['Gesamt (Netzlast) [MWh]', 'Residuallast [MWh]', 'Pumpspeicher [MWh]']
features = ['Monat','Tag', 'Jahr', 'Ist_Wochenende', 'Bevölkerung', 'Temperatur', 'Elektroanteil', 'ImportMWh', 'ExportMWh', 'Ist_Feiertag', 'Wochentag']

# Zielvariable auswählen
selected_variable = st.selectbox("Wähle eine Zielvariable:", variables)

test_year = 2022  # Wähle das Jahr, das als Testdaten verwendet werden soll
# Sicherstellen, dass der Index ein Datumsformat hat
df_all.index = pd.to_datetime(df_all.index)

# Jahr aus dem Index extrahieren und für die Aufteilung verwenden
df_train = df_all[df_all.index.year != test_year]
df_test = df_all[df_all.index.year == test_year]

# Features und Zielvariablen definieren
X_train_raw, y_train = df_train[features], df_train[variables]
X_test_raw, y_test = df_test[features], df_test[variables]

# Skalierung der Zielvariablen
scalers = {}
for var in variables:
    scalers[var] = StandardScaler()
    y_train[[var]] = scalers[var].fit_transform(y_train[[var]])
    y_test[[var]] = scalers[var].transform(y_test[[var]])

# Skalierung der Features
scaler_features = StandardScaler()
X_train_scaled = scaler_features.fit_transform(X_train_raw)
X_test_scaled = scaler_features.transform(X_test_raw)

# Nach der Skalierung die Arrays wieder in DataFrames umwandeln
X_train = pd.DataFrame(X_train_scaled, columns=features, index=X_train_raw.index)
X_test = pd.DataFrame(X_test_scaled, columns=features, index=X_test_raw.index)

# TimeSeriesSplit für Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# Modelle für jede Zielvariable trainieren
models = {}
for variable in variables:
    
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=0.1,
        random_state=50
    )
        
    # Cross-Validation für Evaluierung
    cv_results = []
    for train_idx, val_idx in tscv.split(X_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train[variable].iloc[train_idx], y_train[variable].iloc[val_idx]
        model.fit(X_cv_train, y_cv_train, eval_set=[(X_cv_val, y_cv_val)], verbose=False)
        cv_results.append(model.score(X_cv_val, y_cv_val))
    
    print(f"Cross-Validation Score for {variable}: {np.mean(cv_results):.4f}")
    model.fit(X_train, y_train[variable])
    models[variable] = model

# Vorhersage für 2023 (Testdaten)
predictions_2023 = {}
for variable in variables:
    predictions_2023[variable] = models[variable].predict(X_test)
    
    # Rückskalierung der Vorhersagen
    predictions_2023[variable] = scalers[variable].inverse_transform(predictions_2023[variable].reshape(-1, 1)).flatten()

# Erstelle DataFrame für Vorhersagen
predictions_df_2023 = pd.DataFrame(index=X_test.index)
for variable in variables:
    predictions_df_2023[f"Vorhersage_{variable}"] = predictions_2023[variable]
############################################################################# Visualisierungen

# Berechnung der Performance-Metriken (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Berechnung des MAPE für jedes Modell und Ausgabe in Streamlit
mape_values_2023 = {}
for variable in variables:
    mape = mean_absolute_percentage_error(saved_df_test[variable], predictions_2023[variable])
    mape_values_2023[variable] = mape

st.write("MAPE pro Zielvariable für Testdaten 2023:", mape_values_2023)

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
test_data = saved_df_test[[selected_variable]]
st.write(test_data)
st.write(predictions_2023)
test_data['Vorhersage'] = predictions_2023[selected_variable]
st.write(test_data)
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

# Erstelle DataFrame mit tatsächlichen und vorhergesagten Werten
results_2023 = pd.DataFrame(index=saved_df_test.index)

for variable in variables:
    results_2023[f"Tatsächlich_{variable}"] = saved_df_test[variable]
    results_2023[f"Vorhersage_{variable}"] = predictions_2023[variable]

# Ausgabe des DataFrames in Streamlit
st.write("Vergleich der tatsächlichen und vorhergesagten Werte für Testdaten 2023:", results_2023)