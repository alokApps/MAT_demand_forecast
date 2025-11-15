import pandas as pd
import pmdarima as pm
from prophet import Prophet
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm # For a nice progress bar
import warnings


# Suppress warnings from Prophet and other libraries
warnings.filterwarnings("ignore")

# --- Configuration ---
ORDERS_FILE = 'mat_case_study.xlsx'
POS_FILE = 'mat_case_study.xlsx'
INVENTORY_FILE = 'mat_case_study.xlsx'

FORECAST_STEPS = 6 # Forecast for the next 6 months
POS_LAG = 6 # Use a 6-month lag for POS data as a feature
VALIDATION_MONTHS = 6 # Hold out the last 6 months for validation

# --- Policy Feature Configuration ---
POLICY_START_DATE = '2025-02-01'

OUTPUT_FORECAST_CSV = 'demand_forecast_best_model_next_6_months.csv'
OUTPUT_PLAN_CSV = 'production_plan_best_model_next_6_months.csv'
OUTPUT_VALIDATION_PLOT_DIR = 'validation_plots'
# This is the new main validation scorecard
OUTPUT_OVERALL_VALIDATION_RESULTS_CSV = 'validation_overall_model_comparison.csv'

# --- End Configuration ---

def load_and_preprocess(file_path,sheet_name, value_name='Value'):
    """
    Loads a wide-format CSV, melts it into a long format,
    and converts date strings into proper datetime objects.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    df_long = df.melt(id_vars=['Material'], 
                      var_name='DateStr', 
                      value_name=value_name)
    
    try:
        df_long['Date'] = pd.to_datetime(df_long['DateStr'], format='%y-%b')
    except ValueError as e:
        print(f"Error parsing dates in {file_path}: {e}")
        return None
        
    df_long = df_long.drop(columns=['DateStr']).set_index('Date').sort_index()
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors='coerce')
    
    print(f"Loaded and preprocessed {file_path} (Sheet: {sheet_name})")
    return df_long

def get_rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def plot_sku_validation(sku, history, y_val, 
                        sarima_uni_pred, ets_uni_pred, prophet_uni_pred, 
                        sarimax_exog_pred, prophet_exog_pred):
    """
    Saves a plot of the validation results showing history, actuals,
    and all five competing model forecasts.
    """
    # Calculate RMSE for labels
    sarima_uni_rmse = get_rmse(y_val, sarima_uni_pred)
    ets_uni_rmse = get_rmse(y_val, ets_uni_pred)
    prophet_uni_rmse = get_rmse(y_val, prophet_uni_pred)
    sarimax_exog_rmse = get_rmse(y_val, sarimax_exog_pred)
    prophet_exog_rmse = get_rmse(y_val, prophet_exog_pred)

    plt.figure(figsize=(15, 8))
    
    # Plot history (training data)
    plt.plot(history.index, history, label='Historical Orders (Train)', color='blue', linewidth=1.5)
    
    # Plot actual validation data
    plt.plot(y_val.index, y_val, label='Actual Orders (Validation)', color='black', linewidth=2, marker='o', markersize=6)
    
    # Plot Univariate Model forecasts (dotted)
    plt.plot(y_val.index, sarima_uni_pred, label=f'SARIMA [Uni] (RMSE: {sarima_uni_rmse:.2f})', color='green', linestyle=':')
    plt.plot(y_val.index, ets_uni_pred, label=f'ETS [Uni] (RMSE: {ets_uni_rmse:.2f})', color='purple', linestyle=':')
    plt.plot(y_val.index, prophet_uni_pred, label=f'Prophet [Uni] (RMSE: {prophet_uni_rmse:.2f})', color='red', linestyle=':')
    
    # Plot Exogenous Model forecasts (dashed)
    exog_label = f'(Exog w/ Lag {POS_LAG})'
    plt.plot(y_val.index, sarimax_exog_pred, label=f'SARIMAX {exog_label} (RMSE: {sarimax_exog_rmse:.2f})', color='green', linestyle='--')
    plt.plot(y_val.index, prophet_exog_pred, label=f'Prophet {exog_label} (RMSE: {prophet_exog_rmse:.2f})', color='red', linestyle='--')

    # Plot last 2 years of history + validation for clarity
    plot_start_date = history.index[0]
    if len(history) > 24:
        plot_start_date = history.index[-24] 
    plt.xlim(left=plot_start_date)
    
    plt.title(f'5-Way Model Validation for SKU: {sku}')
    plt.xlabel('Date')
    plt.ylabel('Orders')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_filename = os.path.join(OUTPUT_VALIDATION_PLOT_DIR, f'validation_plot_{sku}.png')
    plt.savefig(plot_filename)
    plt.close() # Close the figure to save memory

def run_forecasting_loop(df_orders, df_pos, forecast_steps, pos_lag):
    """
    Loops through each SKU, validates 5 models (3 Uni, 2 Exog),
    selects the single best model, and forecasts.
    """
    if df_orders is None:
        return None

    unique_skus = sorted(list(set(df_orders['Material'].unique()) | set(df_pos['Material'].unique())))
    all_forecasts = []
    all_validation_results = []
    
    if not os.path.exists(OUTPUT_VALIDATION_PLOT_DIR):
        os.makedirs(OUTPUT_VALIDATION_PLOT_DIR)

    print(f"Starting forecast for {len(unique_skus)} SKUs...")

    for sku in tqdm(unique_skus, desc="Forecasting SKUs"):
        
        # --- 1. Prepare SKU-level data ---
        sku_orders = df_orders[df_orders['Material'] == sku]['Orders'].asfreq('MS')
        sku_pos = df_pos[df_pos['Material'] == sku]['POS'].asfreq('MS')
        
        combined_df = pd.DataFrame({'Orders': sku_orders})
        combined_df = combined_df.join(sku_pos.rename('POS'), how='outer')
        
        # Interpolate and fill
        combined_df['Orders'] = combined_df['Orders'].interpolate(method='time').fillna(0)
        combined_df['POS'] = combined_df['POS'].interpolate(method='time').fillna(0)

        # --- 2. Feature Engineering (Exogenous Variables) ---
        exog_col_name = f'POS_lag_{pos_lag}'
        combined_df[exog_col_name] = combined_df['POS'].shift(pos_lag)
        
        policy_start = pd.to_datetime(POLICY_START_DATE)
        combined_df['Policy_Active'] = (combined_df.index >= policy_start).astype(int)
        
        final_df = combined_df.dropna(subset=['Orders', exog_col_name]) # Drop NAs from shifting
        
        # Check for sufficient data
        if len(final_df) < (VALIDATION_MONTHS + 24):
            print(f"Skipping SKU {sku}: Insufficient data for validation ({len(final_df)} months). Need at least {VALIDATION_MONTHS + 24}.")
            continue

        # --- 3. Split Data for Validation ---
        train_end_index = len(final_df) - VALIDATION_MONTHS
        train_df = final_df.iloc[:train_end_index]
        validation_df = final_df.iloc[train_end_index:]

        y_train = train_df['Orders']
        y_val = validation_df['Orders']
        
        # Create exogenous variable sets
        exog_features = [exog_col_name, 'Policy_Active']
        X_train_exog = train_df[exog_features]
        X_val_exog = validation_df[exog_features]
        
        
        # --- 4. MODEL BAKE-OFF ---
        
        # --- Model 1: SARIMA (Univariate) ---
        try:
            sarima_uni_model = pm.auto_arima(y_train, 
                                          X=None, # No exogenous features
                                          seasonal=True, m=12,
                                          suppress_warnings=True,
                                          stepwise=True,
                                          error_action='ignore')
            
            sarima_uni_pred = sarima_uni_model.predict(n_periods=VALIDATION_MONTHS, X=None)
            sarima_uni_pred = pd.Series(sarima_uni_pred, index=y_val.index).clip(lower=0)
            sarima_uni_error = get_rmse(y_val, sarima_uni_pred)
        except Exception as e:
            print(f"SARIMA (Uni) failed for {sku}: {e}")
            sarima_uni_error = np.inf
            sarima_uni_pred = pd.Series([0]*VALIDATION_MONTHS, index=y_val.index)

        # --- Model 2: ETS (Univariate) ---
        try:
            ets_uni_model = ExponentialSmoothing(
                y_train.astype(float), # .astype(float) for stability
                seasonal='add', 
                seasonal_periods=12
            ).fit()
            ets_uni_pred = ets_uni_model.forecast(VALIDATION_MONTHS)
            ets_uni_pred = pd.Series(ets_uni_pred, index=y_val.index).clip(lower=0)
            ets_uni_error = get_rmse(y_val, ets_uni_pred)
        except Exception as e:
            print(f"ETS (Uni) failed for {sku}: {e}")
            ets_uni_error = np.inf
            ets_uni_pred = pd.Series([0]*VALIDATION_MONTHS, index=y_val.index)

        # --- Model 3: Prophet (Univariate) ---
        try:
            prophet_train_df_uni = pd.DataFrame({'ds': y_train.index, 'y': y_train.values})

            prophet_uni_model = Prophet(yearly_seasonality=True)
            prophet_uni_model.fit(prophet_train_df_uni) # mcmc_samples=0 for speed

            future_df_uni = pd.DataFrame({'ds': validation_df.index})
            
            prophet_forecast_uni = prophet_uni_model.predict(future_df_uni)
            prophet_uni_pred = prophet_forecast_uni['yhat'].clip(lower=0).values
            prophet_uni_pred = pd.Series(prophet_uni_pred, index=y_val.index)
            prophet_uni_error = get_rmse(y_val, prophet_uni_pred)
        except Exception as e:
            print(f"Prophet (Uni) failed for {sku}: {e}")
            prophet_uni_error = np.inf
            prophet_uni_pred = pd.Series([0]*VALIDATION_MONTHS, index=y_val.index)

        # --- Model 4: SARIMAX (Exogenous) ---
        try:
            sarimax_exog_model = pm.auto_arima(y_train, 
                                          X=X_train_exog, # <-- Use Exog features
                                          seasonal=True, m=12,
                                          suppress_warnings=True,
                                          stepwise=True,
                                          error_action='ignore')
            
            sarimax_exog_pred = sarimax_exog_model.predict(n_periods=VALIDATION_MONTHS, X=X_val_exog)
            sarimax_exog_pred = pd.Series(sarimax_exog_pred, index=y_val.index).clip(lower=0)
            sarimax_exog_error = get_rmse(y_val, sarimax_exog_pred)
        except Exception as e:
            print(f"SARIMAX (Exog) failed for {sku}: {e}")
            sarimax_exog_error = np.inf
            sarimax_exog_pred = pd.Series([0]*VALIDATION_MONTHS, index=y_val.index)

        # --- Model 5: Prophet (Exogenous) ---
        try:
            prophet_train_df_exog = pd.DataFrame({'ds': y_train.index, 'y': y_train.values})
            prophet_train_df_exog[exog_col_name] = X_train_exog[exog_col_name].values
            prophet_train_df_exog['Policy_Active'] = X_train_exog['Policy_Active'].values

            prophet_exog_model = Prophet(yearly_seasonality=True)
            prophet_exog_model.add_regressor(exog_col_name)
            prophet_exog_model.add_regressor('Policy_Active')
            prophet_exog_model.fit(prophet_train_df_exog) # mcmc_samples=0 for speed

            future_df_exog = pd.DataFrame({'ds': validation_df.index})
            future_df_exog[exog_col_name] = X_val_exog[exog_col_name].values
            future_df_exog['Policy_Active'] = X_val_exog['Policy_Active'].values
            
            prophet_forecast_exog = prophet_exog_model.predict(future_df_exog)
            prophet_exog_pred = prophet_forecast_exog['yhat'].clip(lower=0).values
            prophet_exog_pred = pd.Series(prophet_exog_pred, index=y_val.index)
            prophet_exog_error = get_rmse(y_val, prophet_exog_pred)
        except Exception as e:
            print(f"Prophet (Exog) failed for {sku}: {e}")
            prophet_exog_error = np.inf
            prophet_exog_pred = pd.Series([0]*VALIDATION_MONTHS, index=y_val.index)
            
        # --- 5. Plot Validation Results ---
        plot_sku_validation(sku, y_train, y_val, 
                            sarima_uni_pred, ets_uni_pred, prophet_uni_pred, 
                            sarimax_exog_pred, prophet_exog_pred)
        
        # --- 6. Compare and Select Best Overall Model ---
        errors = {
            'SARIMA_Uni': sarima_uni_error,
            'ETS_Uni': ets_uni_error,
            'Prophet_Uni': prophet_uni_error,
            'SARIMAX_Exog': sarimax_exog_error,
            'Prophet_Exog': prophet_exog_error
        }
        
        # Find the model with the minimum RMSE
        best_overall_model_name = min(errors, key=errors.get)
        best_overall_error = errors[best_overall_model_name]

        all_validation_results.append({
            'Material': sku,
            'Best_Overall_Model': best_overall_model_name,
            'Best_RMSE': best_overall_error,
            'SARIMA_Uni_RMSE': sarima_uni_error,
            'ETS_Uni_RMSE': ets_uni_error,
            'Prophet_Uni_RMSE': prophet_uni_error,
            'SARIMAX_Exog_RMSE': sarimax_exog_error,
            'Prophet_Exog_RMSE': prophet_exog_error
        })

        # --- 7. Final Forecast (with Best Overall Model) ---
        
        # Prepare full dataset for re-training
        y_full = final_df['Orders']
        X_full_exog = final_df[exog_features]
        
        # Prepare future exogenous data
        # We need the *last* known POS values to shift forward for the forecast
        future_pos_inputs = combined_df['POS'].iloc[-pos_lag:].values 

        forecast_dates = pd.date_range(start=y_full.index[-1] + pd.DateOffset(months=1), 
                                     periods=forecast_steps, 
                                     freq='MS')
        
        X_future_exog = pd.DataFrame(index=forecast_dates)
        
        # Create future lagged values. 
        future_lag_values = np.zeros(forecast_steps)
        for i in range(forecast_steps):
            if i < len(future_pos_inputs):
                future_lag_values[i] = future_pos_inputs[i]
            else:
                # If forecast steps > pos_lag, we've run out of "known" POS data
                # We'll just use the last known value (a simple assumption)
                future_lag_values[i] = future_pos_inputs[-1] 

        X_future_exog[exog_col_name] = future_lag_values[:forecast_steps]
        X_future_exog['Policy_Active'] = 1 # Policy is active for all future dates

        try:
            if best_overall_model_name == 'SARIMA_Uni':
                final_model = pm.auto_arima(y_full, X=None, seasonal=True, m=12,
                                            suppress_warnings=True, stepwise=True, error_action='ignore')
                final_forecast_values = final_model.predict(n_periods=forecast_steps, X=None)
            
            elif best_overall_model_name == 'ETS_Uni':
                final_model = ExponentialSmoothing(y_full.astype(float), seasonal='add', seasonal_periods=12).fit()
                final_forecast_values = final_model.forecast(forecast_steps)

            elif best_overall_model_name == 'Prophet_Uni':
                prophet_full_df_uni = pd.DataFrame({'ds': y_full.index, 'y': y_full.values})
                final_model = Prophet(yearly_seasonality=True).fit(prophet_full_df_uni)
                future_df_pred_uni = pd.DataFrame({'ds': forecast_dates})
                final_forecast = final_model.predict(future_df_pred_uni)
                final_forecast_values = final_forecast['yhat'].values

            elif best_overall_model_name == 'SARIMAX_Exog':
                final_model = pm.auto_arima(y_full, X=X_full_exog, seasonal=True, m=12,
                                            suppress_warnings=True, stepwise=True, error_action='ignore')
                final_forecast_values = final_model.predict(n_periods=forecast_steps, X=X_future_exog)
            
            else: # Best model is 'Prophet_Exog'
                prophet_full_df_exog = pd.DataFrame({'ds': y_full.index, 'y': y_full.values})
                prophet_full_df_exog[exog_col_name] = X_full_exog[exog_col_name].values
                prophet_full_df_exog['Policy_Active'] = X_full_exog['Policy_Active'].values
                
                final_model = Prophet(yearly_seasonality=True)
                final_model.add_regressor(exog_col_name)
                final_model.add_regressor('Policy_Active')
                final_model.fit(prophet_full_df_exog)

                future_df_pred_exog = pd.DataFrame({'ds': forecast_dates})
                future_df_pred_exog[exog_col_name] = X_future_exog[exog_col_name].values
                future_df_pred_exog['Policy_Active'] = X_future_exog['Policy_Active'].values
                
                final_forecast = final_model.predict(future_df_pred_exog)
                final_forecast_values = final_forecast['yhat'].values

        except Exception as e:
            print(f"Final forecast failed for {sku} (model {best_overall_model_name}): {e}")
            final_forecast_values = [0] * forecast_steps
        
        # --- 8. Store Final Forecast ---
        final_forecast_values = [max(0, round(val)) for val in final_forecast_values]
        
        sku_forecast_df = pd.DataFrame({
            'Material': sku,
            'Date': forecast_dates,
            'Forecast_Orders': final_forecast_values
        })
        all_forecasts.append(sku_forecast_df)
    
    print("Forecasting loop complete.")
    
    # Save validation results
    pd.DataFrame(all_validation_results).to_csv(OUTPUT_OVERALL_VALIDATION_RESULTS_CSV, index=False)
    
    if not all_forecasts:
        print("No forecasts were generated.")
        return None
        
    all_forecasts_df = pd.concat(all_forecasts)
    
    try:
        forecast_pivot = all_forecasts_df.pivot(index='Material', 
                                           columns='Date', 
                                           values='Forecast_Orders')
        forecast_pivot.columns = [col.strftime('%Y-%b') for col in forecast_pivot.columns]
        return forecast_pivot
        
    except Exception as e:
        print(f"Could not pivot results: {e}")
        return None

# --- Functions below are unchanged ---

def generate_production_plan(forecast_pivot_df, df_inventory):
    """
    Uses the forecast and inventory-on-hand to calculate
    net production needs for the next 6 months.
    """
    if forecast_pivot_df is None or df_inventory is None:
        print("Skipping production plan generation due to missing data.")
        return

    print("Generating production plan...")
    
    # Merge forecast with current inventory
    plan_df = forecast_pivot_df.merge(df_inventory, on='Material', how='left')
    
    # Fill missing inventory with 0
    plan_df['Inventory_On_Hand'] = plan_df['Inventory_On_Hand'].fillna(0)
    
    # Re-order columns to be more logical
    if forecast_pivot_df.empty:
        print("Forecast pivot is empty. Cannot generate production plan.")
        return

    forecast_cols = list(forecast_pivot_df.columns)
    plan_df = plan_df[['Material', 'Inventory_On_Hand'] + forecast_cols]
    
    # This will store our new calculated columns
    new_cols_df = pd.DataFrame(index=plan_df.index)
    
    # Get the current inventory
    current_inventory = plan_df['Inventory_On_Hand']
    
    # Loop through each month of the forecast
    for col in forecast_cols:
        forecast_demand = plan_df[col]
        
        # Calculate net need: demand - inventory, but not less than 0
        net_need = (forecast_demand - current_inventory).clip(lower=0)
        
        # Calculate ending inventory: inventory - demand, but not less than 0
        ending_inventory = (current_inventory - forecast_demand).clip(lower=0)
        
        # Store results
        new_cols_df[f'Net_Need_{col}'] = net_need.round()
        new_cols_df[f'Ending_Inv_{col}'] = ending_inventory.round()
        
        # The ending inventory of this month is the *starting* inventory for the next
        current_inventory = ending_inventory

    # Combine the original data with the new calculated columns
    final_plan_df = pd.concat([plan_df, new_cols_df], axis=1)
    
    # Save the plan
    final_plan_df.to_csv(OUTPUT_PLAN_CSV, index=False)
    print(f"Successfully saved production plan to {OUTPUT_PLAN_CSV}")


def main():
    """Main execution function."""
    # --- 1. Load All Data Sources ---
    df_orders = load_and_preprocess(ORDERS_FILE,'Actual Orders', 'Orders')
    df_pos = load_and_preprocess(POS_FILE,'POS', 'POS')
    
    try:
        df_inventory = pd.read_excel(INVENTORY_FILE, sheet_name="Inventory")
    except FileNotFoundError:
        print(f"Error: Inventory file not found at {INVENTORY_FILE}")
        df_inventory = None

    # --- 2. Run Forecasting & Validation Loop ---
    forecast_pivot_df = run_forecasting_loop(df_orders, 
                                             df_pos, 
                                             FORECAST_STEPS, 
                                             POS_LAG)
    
    if forecast_pivot_df is not None:
        # Save the raw forecast file
        forecast_pivot_df.to_csv(OUTPUT_FORECAST_CSV)
        print(f"\nSuccessfully saved 6-month 'best model' forecast to {OUTPUT_FORECAST_CSV}")
        print(f"Model comparison results saved to {OUTPUT_OVERALL_VALIDATION_RESULTS_CSV}")
        print(f"Individual SKU validation plots saved to '{OUTPUT_VALIDATION_PLOT_DIR}'")

        # --- 3. Generate Production Plan ---
        generate_production_plan(forecast_pivot_df, df_inventory)
    
    else:
        print("No forecast was generated. Exiting.")


if __name__ == "__main__":
    print("Starting 5-Way Model Validation & Forecasting Process ...")
    main()
    print("Process Finished.")