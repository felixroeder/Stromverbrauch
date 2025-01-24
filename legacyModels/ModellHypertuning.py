# Zielvariablen und Features
variables = ['Gesamt (Netzlast) [MWh]', 'Residuallast [MWh]', 'Pumpspeicher [MWh]']
features = ['Tag', 'Monat', 'Jahr', 'Ist_Wochenende', 'Temperatur', 'Ist_Feiertag', 'Wochentag']

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

# Definiere den Hyperparameter-Raster
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Modelle für jede Zielvariable trainieren
models = {}
for variable in variables:
    best_score = float('inf')
    best_model = None
    best_params = None
    
    # Durchlaufe alle möglichen Hyperparameter-Kombinationen
    for params in ParameterGrid(param_grid):
        model = XGBRegressor(objective='reg:squarederror', random_state=50, **params)

        # Cross-Validation mit TimeSeriesSplit
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train[variable].iloc[train_idx], y_train[variable].iloc[val_idx]
            
            model.fit(X_cv_train, y_cv_train)
            predictions = model.predict(X_cv_val)
            score = mean_squared_error(y_cv_val, predictions)
            cv_scores.append(score)
        
        # Berechne den durchschnittlichen Fehler über alle Cross-Validation-Splits
        mean_cv_score = np.mean(cv_scores)
        print(f"Hyperparameters: {params}, CV MSE: {mean_cv_score}")
        
        # Wenn das aktuelle Modell besser ist, speichere es
        if mean_cv_score < best_score:
            best_score = mean_cv_score
            best_model = model
            best_params = params
    
    # Trainiere das Modell mit den besten Parametern auf den gesamten Trainingsdaten
    print(f"Best Hyperparameters for {variable}: {best_params}")
    best_model.fit(X_train, y_train[variable])
    
    # Modell speichern
    models[variable] = best_model

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