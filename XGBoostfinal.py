import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib  # Für das Speichern und Laden der Modelle

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

# Modellpfad
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

######################################################## DATENVORBEREITUNG ###########################################################################################################
######################################################## DATENVORBEREITUNG ###########################################################################################################

#################################### Trainingsdaten
# Datenpfade
pathHistorisch = 'Data/Regionsdaten 17-22'
weather_data_path = "Data/Wetterdaten.xlsx"
holiday_data_path = "Data/FeiertageDE.xlsx"
einwohner_data_path = "Data/Bevölkerung17-22.xlsx"
elektroKraftwagen_data_path = "Data/NeuzulassungenKraftfahrt17-22.xlsx"
importexport_data_path = "Data/ExportImport17-22.xlsx"

files = [f for f in os.listdir(pathHistorisch) if f.endswith('.xlsx')]

regions = []
data_by_region = {}

# Für jede Region die Daten einlesen
for file in files:
    region_name = os.path.splitext(file)[0]
    regions.append(region_name)
    file_path = os.path.join(pathHistorisch, file)
    df = pd.read_excel(file_path, engine='openpyxl')
    df['Region'] = region_name
    data_by_region[region_name] = df

# Wetterdaten einlesen
weather_df = pd.read_excel(weather_data_path, engine='openpyxl')
weather_df.columns = ['Station', 'Datum', 'Temperatur']
weather_df = weather_df.groupby('Datum')['Temperatur'].mean().reset_index()

# Feiertage einlesen
holiday_df = pd.read_excel(holiday_data_path, engine='openpyxl')
holiday_df.columns = ['Datum']

# Bevölkerung einlesen
einwohner_df = pd.read_excel(einwohner_data_path, engine='openpyxl')

# Reine Elekrokraftwagen Neuzulassungen in % einlesen (monatliche Basis)
elektroKraftwagen_df = pd.read_excel(elektroKraftwagen_data_path, engine='openpyxl')

# Import und Export von Strom (monatliche Basis)
importexport_df = pd.read_excel(importexport_data_path, engine='openpyxl')

# Datensätze kombinieren
df_all = pd.concat(data_by_region.values(), ignore_index=True)
df_all = df_all.merge(weather_df, on='Datum', how='left')
df_all = df_all.merge(einwohner_df, on='Datum', how='left')
df_all = df_all.merge(elektroKraftwagen_df, on='Datum', how='left')
df_all = df_all.merge(importexport_df, on='Datum', how='left')

# Konvertiere Datumsspalte zu Datetime und setze als Index
df_all['Datum'] = pd.to_datetime(df_all['Datum'], errors='coerce')
df_all.set_index('Datum', inplace=True)

# Zeilen mit leeren Werten entfernen
df_all = df_all.dropna()

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
data_path_test = 'Data/Regionsdaten 23'

files_test = [f for f in os.listdir(data_path_test) if f.endswith('.xlsx')]

regions_test = []
data_by_region_test = {}

# Daten pro Region laden
for file in files_test:
    region_name = os.path.splitext(file)[0]  # Region aus Dateinamen ableiten
    regions_test.append(region_name)
    file_path = os.path.join(data_path_test, file)
    df = pd.read_excel(file_path, engine='openpyxl')
    df['Region'] = region_name
    data_by_region_test[region_name] = df

# Regionale Daten kombinieren
df_all_test = pd.concat(data_by_region_test.values(), ignore_index=True)

# Konvertiere Datumsspalte zu Datetime und setze als Index
df_all_test['Datum'] = pd.to_datetime(df_all_test['Datum'], errors='coerce')
df_all_test.set_index('Datum', inplace=True)

# Zeilen mit leeren Werten entfernen
df_all_test = df_all_test.dropna()

# Regionen umbenennen (23 weg)
df_all_test.loc[df_all_test['Region'] == 'Amprion23', 'Region'] = 'Amprion'
df_all_test.loc[df_all_test['Region'] == '50Hertz23', 'Region'] = '50Hertz'
df_all_test.loc[df_all_test['Region'] == 'TenneT23', 'Region'] = 'TenneT'
df_all_test.loc[df_all_test['Region'] == 'TransnetBW23', 'Region'] = 'TransnetBW'

saved_df_test = df_all_test.copy()

######################################################## MODELL ###########################################################################################################
######################################################## MODELL ###########################################################################################################

# Zielvariablen und Features
variables = ['Gesamt (Netzlast) [MWh]', 'Residuallast [MWh]', 'Pumpspeicher [MWh]']
features = ['Monat','Tag', 'Jahr', 'Ist_Wochenende', 'Bevölkerung', 'Temperatur', 'Elektroanteil', 'ImportMWh', 'ExportMWh', 'Ist_Feiertag', 'Wochentag']

test_year = 2022  # Testjahr
regions = df_all['Region'].unique()  # Einzigartige Regionen
final_predictions_file = os.path.join(model_dir, "final_predictions.pkl")  # Speicherort für den finalen Datensatz
all_predictions = []

# Sicherstellen, dass der Index ein Datumsformat hat
df_all.index = pd.to_datetime(df_all.index)

# Überprüfen, ob der finale Datensatz bereits existiert
if os.path.exists(final_predictions_file):
    # Wenn der Datensatz existiert, lade ihn
    print(f"Final predictions already exist. Loading predictions from {final_predictions_file}.")
    final_predictions_df = joblib.load(final_predictions_file)
else:
    # Iterieren über jede Region und Modelltraining durchführen, wenn der finale Datensatz nicht existiert
    for region in regions:
        print(f"Processing Region: {region}")
        
        # Daten filtern nach Region
        df_region = df_all[df_all['Region'] == region]
        
        # Training- und Test-Daten aufteilen
        df_train = df_region[df_region.index.year != test_year]
        df_test = df_region[df_region.index.year == test_year]
        
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
        
        # Arrays in DataFrames umwandeln
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
            
            # Cross-Validation
            cv_results = []
            for train_idx, val_idx in tscv.split(X_train):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train[variable].iloc[train_idx], y_train[variable].iloc[val_idx]
                model.fit(X_cv_train, y_cv_train, eval_set=[(X_cv_val, y_cv_val)], verbose=False)
                cv_results.append(model.score(X_cv_val, y_cv_val))
            
            print(f"Cross-Validation Score for {variable} in Region {region}: {np.mean(cv_results):.4f}")
            model.fit(X_train, y_train[variable])
            models[variable] = model
        
        # Vorhersagen für die Testdaten
        predictions = {}
        for variable in variables:
            predictions[variable] = models[variable].predict(X_test)
            
            # Rückskalierung der Vorhersagen
            predictions[variable] = scalers[variable].inverse_transform(predictions[variable].reshape(-1, 1)).flatten()

        # DataFrame für die aktuelle Region erstellen
        predictions_df = pd.DataFrame(index=X_test.index)
        for variable in variables:
            predictions_df[variable] = predictions[variable]
        
        # Region als Spalte hinzufügen
        predictions_df["Region"] = region
        
        # DataFrame in die Liste hinzufügen
        all_predictions.append(predictions_df)

    # Alle Regionen in einen einzigen Datensatz zusammenfassen
    final_predictions_df = pd.concat(all_predictions)

    # Index zurücksetzen
    final_predictions_df.reset_index(inplace=True)

    final_predictions_df = final_predictions_df.drop(columns=['Datum'])
    
    # Speichern des finalen Datensatzes
    joblib.dump(final_predictions_df, final_predictions_file)

######################################################## VISUALISIERUNGEN ###########################################################################################################
######################################################## VISUALISIERUNGEN ###########################################################################################################

# Wochentagsspalte entfernen - brauchen wir nicht für die Visualisierung
saved_df_test = saved_df_test.drop(columns=['Wochentag'])

# Sicherstellen, dass 'Datum' korrekt als Index vorhanden ist
if 'Datum' not in saved_df_test.columns:
    saved_df_test.reset_index(inplace=True)  # Falls 'Datum' der Index war, zurücksetzen

# Anzuzeigende Seite auswählen
page = st.sidebar.selectbox(
    "Wähle eine Seite:",
    ["Vorhersagedaten 2023", "Vergleich mit echten Daten"]
)

#################################### Seite 1
if page == "Vorhersagedaten 2023":
    # Überschrift
    st.write("Alle Angaben in MWh")
    st.markdown(f"<h2 style='text-align: center;'>Stromverbrauchsdaten für das Jahr 2023</h2>", unsafe_allow_html=True)
    st.markdown("")

    # Region und Zielvariable auswählen
    selected_region = st.selectbox("Wähle eine Region", regions)
    selected_variable = st.selectbox("Wähle eine Zielvariable", variables)

    # Filter auf die Daten anwenden
    filtered_test_df = (
        saved_df_test[saved_df_test['Region'] == selected_region]
        .loc[:, ['Datum', selected_variable]]
        .rename(columns={selected_variable: "Tatsächlich"})
    )
    filtered_predictions_df = (
        final_predictions_df[final_predictions_df['Region'] == selected_region]
        .loc[:, [selected_variable]]
        .rename(columns={selected_variable: "Vorhersage"})
    )

    filtered_test_df.reset_index(drop=True, inplace=True)  # Index zurücksetzen
    filtered_predictions_df.reset_index(drop=True, inplace=True)  # Index zurücksetzen

    # Wenn 'Datum' im Index ist, müssen wir es zurück in eine Spalte umwandeln
    filtered_test_df.reset_index(inplace=True)

    # Datum in datetime-Format umwandeln
    filtered_test_df['Datum'] = pd.to_datetime(filtered_test_df['Datum'])

    # Index der Vorhersagen auf Datum setzen
    filtered_predictions_df['Datum'] = filtered_test_df['Datum']

    # Fertiger Datensatz für die Visualisierung auf Basis der ausgewählten Filter
    combined_data = pd.merge(filtered_test_df, filtered_predictions_df, on='Datum')

    # Datum als Index setzen
    combined_data.set_index('Datum', inplace=True)
    combined_data = combined_data.drop(columns=['index'])
    combined_data = combined_data.drop(columns=['Tatsächlich']) # Brauche ich nicht auf dieser Seite - kein Vergleich mit echten Daten

    # Datensatz mit allen Regionen
    all_regions = (
        final_predictions_df.loc[:, ['Region', selected_variable]]
        .rename(columns={selected_variable: "Vorhersage"})
    )

    # Zeitraum einschränken
    start_date, end_date = st.date_input(
        "Wähle den Betrachtungszeitraum",
        value=(combined_data.index.min(), combined_data.index.max()),
        min_value=combined_data.index.min(),
        max_value=combined_data.index.max(),
    )

    # Filterung des Datensatzes
    combined_data = combined_data.loc[start_date:end_date]

    st.markdown("---") ##################### TRENNLINIE

    # Durschnittswert und absoluter Wert
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<h5 style='text-align: center;'>Durchschnittswert</h5>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>{combined_data['Vorhersage'].mean().round(2)}</h3>", unsafe_allow_html=True)
    with col2: 
        st.markdown(f"<h5 style='text-align: center;'>Absoluter Wert</h5>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>{combined_data['Vorhersage'].sum().round(2)}</h3>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown(f"<h5 style='text-align: center;'>Zeitverlauf</h5>", unsafe_allow_html=True)
    st.line_chart(combined_data)

    st.markdown("---") ##################### TRENNLINIE

    st.markdown(f"<h5 style='text-align: center;'>Verteilung nach Region - unabhängig von Zeitraum</h5>", unsafe_allow_html=True)

    # Summen für jede Region berechnen und in einer Liste speichern
    wertefuerVergleich = []

    for region in all_regions['Region'].unique():  # Einmalig jede Region durchlaufen
        region_sum = all_regions[all_regions['Region'] == region]['Vorhersage'].sum()
        wertefuerVergleich.append({'Region': region, 'Summe': region_sum})

    # In einen DataFrame umwandeln für die Visualisierung
    vergleich_df = pd.DataFrame(wertefuerVergleich)

    # Bar Chart erstellen
    st.bar_chart(vergleich_df.set_index('Region')['Summe'])

    st.markdown("---") ##################### TRENNLINIE

    st.markdown(f"<h5 style='text-align: center;'>Identifizierung von Spitzen</h5>", unsafe_allow_html=True)

    # Durschnittswert und absoluter Wert
    col1, col2 = st.columns(2)

    with col1:
        st.write("5 Tage mit den höchsten Vorhersagen")
        st.table(combined_data.nlargest(5, 'Vorhersage').round(2))

    with col2:
        st.write("5 Tage mit den niedrigsten Vorhersagen")
        st.table(combined_data.nsmallest(5, 'Vorhersage').round(2))


################### Seite 2
if page == "Vergleich mit echten Daten":
    # Überschrift
    st.write("Alle Angaben in MWh")
    st.markdown(f"<h2 style='text-align: center;'>Vergleich mit echten Verbrauchsdaten von 2023</h2>", unsafe_allow_html=True)
    st.markdown("")

    # Region und Zielvariable auswählen
    selected_region = st.selectbox("Wähle eine Region", regions)
    selected_variable = st.selectbox("Wähle eine Zielvariable", variables)

    # Filter auf die Daten anwenden
    filtered_test_df = (
        saved_df_test[saved_df_test['Region'] == selected_region]
        .loc[:, ['Datum', selected_variable]]
        .rename(columns={selected_variable: "Tatsächlich"})
    )
    filtered_predictions_df = (
        final_predictions_df[final_predictions_df['Region'] == selected_region]
        .loc[:, [selected_variable]]
        .rename(columns={selected_variable: "Vorhersage"})
    )

    filtered_test_df.reset_index(drop=True, inplace=True)  # Index zurücksetzen
    filtered_predictions_df.reset_index(drop=True, inplace=True)  # Index zurücksetzen

    # Wenn 'Datum' im Index ist, müssen wir es zurück in eine Spalte umwandeln
    filtered_test_df.reset_index(inplace=True)

    # Datum in datetime-Format umwandeln
    filtered_test_df['Datum'] = pd.to_datetime(filtered_test_df['Datum'])

    # Index der Vorhersagen auf Datum setzen
    filtered_predictions_df['Datum'] = filtered_test_df['Datum']

    # Fertiger Datensatz für die Visualisierung auf Basis der ausgewählten Filter
    combined_data = pd.merge(filtered_test_df, filtered_predictions_df, on='Datum')

    # Datum als Index setzen
    combined_data.set_index('Datum', inplace=True)
    combined_data = combined_data.drop(columns=['index'])

    st.markdown("---") ##################### TRENNLINIE - MAPE

    # Berechnung der Performance-Metriken (MAPE) für die ausgewählte Zielvariable
    st.markdown(f"<h5 style='text-align: center;'>MAPE-Wert für die Region {selected_region} mit der Zielvariable {selected_variable}</h5>", unsafe_allow_html=True)

    # Berechnung der MAPE nur für die gewählte Zielvariable
    y_true = combined_data['Tatsächlich']
    y_pred = combined_data['Vorhersage']
    region_mape = mean_absolute_percentage_error(y_true, y_pred)

    st.markdown(f"<h3 style='text-align: center;'>{region_mape*100:.2f}%</h3>", unsafe_allow_html=True)

    st.markdown("---") ##################### TRENNLINIE - Vergleich Daten

    st.markdown(f"<h5 style='text-align: center;'>Vergleich der vorhergesagten Daten mit echten Daten für die Region {selected_region} mit der Zielvariable {selected_variable}</h5>", unsafe_allow_html=True)

    # Datumsbereich-Widget für die Testdaten
    start_date_test, end_date_test = st.date_input(
        "Wähle den Betrachtungszeitraum für das Diagramm",
        value=(combined_data.index.min(), combined_data.index.max()),
        min_value=combined_data.index.min(),
        max_value=combined_data.index.max(),
    )

    # Filterung und Visualisierung der Testdaten
    combined_data_time = combined_data.loc[start_date_test:end_date_test]
    st.write(f"Echte Verbrauchsdaten ({start_date_test} bis {end_date_test}) mit Vorhersagen")

    # Visualisierung der tatsächlichen Werte und der Vorhersagen für die Testdaten
    st.line_chart(combined_data_time[['Tatsächlich', 'Vorhersage']])

    # Resultat anzeigen
    st.write(combined_data)
