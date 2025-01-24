################ MODELL #############################
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

# Modelle für jede Zielvariable trainieren
models = {}
for variable in variables:
    model_path = os.path.join(model_dir, f"xgboost_{variable}.json")
    
    if os.path.exists(model_path):
        # Lade den Booster direkt
        booster = Booster()
        booster.load_model(model_path)
        
        # Verwende den Booster im XGBRegressor
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=42
        )
        model._Booster = booster
    else:
        # Hyperparameter-Tuning
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=42
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
        model.save_model(model_path)
        
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