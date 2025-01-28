# Zielvariablen und Features
variables = ['Gesamt (Netzlast) [MWh]', 'Residuallast [MWh]', 'Pumpspeicher [MWh]']
features = ['Monat', 'Tag', 'Jahr', 'Ist_Wochenende', 'Bevölkerung', 'Temperatur', 'Elektroanteil', 'ImportMWh', 'ExportMWh', 'Ist_Feiertag', 'Wochentag']

test_year = 2022  # Testjahr
regions = df_all['Region'].unique()  # Einzigartige Regionen
final_predictions_file = os.path.join(model_dir, "final_predictions.pkl")  # Speicherort für den finalen Datensatz
all_predictions = []

# Sicherstellen, dass der Index ein Datumsformat hat
df_all.index = pd.to_datetime(df_all.index)

# Definiere den Hyperparameter-Raster
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Überprüfen, ob der finale Datensatz bereits existiert
if os.path.exists(final_predictions_file):
    print(f"Final predictions already exist. Loading predictions from {final_predictions_file}.")
    final_predictions_df = joblib.load(final_predictions_file)
else:
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
            best_score = float('inf')
            best_model = None
            best_params = None
            
            for params in ParameterGrid(param_grid):
                model = XGBRegressor(objective='reg:squarederror', random_state=50, **params)
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_train):
                    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_cv_train, y_cv_val = y_train[variable].iloc[train_idx], y_train[variable].iloc[val_idx]
                    
                    model.fit(X_cv_train, y_cv_train)
                    predictions = model.predict(X_cv_val)
                    score = mean_squared_error(y_cv_val, predictions)
                    cv_scores.append(score)
                
                mean_cv_score = np.mean(cv_scores)
                if mean_cv_score < best_score:
                    best_score = mean_cv_score
                    best_model = model
                    best_params = params
            
            print(f"Best Hyperparameters for {variable} in Region {region}: {best_params}")
            best_model.fit(X_train, y_train[variable])
            models[variable] = best_model
        
        # Vorhersagen für die Testdaten
        predictions = {}
        for variable in variables:
            predictions[variable] = models[variable].predict(X_test)
            predictions[variable] = scalers[variable].inverse_transform(predictions[variable].reshape(-1, 1)).flatten()
        
        # DataFrame für die aktuelle Region erstellen
        predictions_df = pd.DataFrame(index=X_test.index)
        for variable in variables:
            predictions_df[variable] = predictions[variable]
        
        predictions_df["Region"] = region
        all_predictions.append(predictions_df)

    # Alle Regionen in einen einzigen Datensatz zusammenfassen
    final_predictions_df = pd.concat(all_predictions)
    final_predictions_df.reset_index(inplace=True)

    # Speichern des finalen Datensatzes
    joblib.dump(final_predictions_df, final_predictions_file)