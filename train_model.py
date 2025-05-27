import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import pathlib
import os

def train_and_save_model(data_file, output_dir, city_name):
    """Train the neural network model and save it along with the scaler"""
    
    print(f"\n=== Training {city_name} model ===")
    print(f"Using data file: {data_file}")
    print(f"Output directory: {output_dir}")
    
    out_dir = pathlib.Path(output_dir)
    if not out_dir.exists(): 
        out_dir.mkdir(parents=True)
    
    file = pd.read_csv(data_file)
    
    input_vars = ["month", "day", "land_atmosphere", "sea_atmosphere", "precipitation", 
                  "temperature", "humidity", "wind_speed", "wind_direction", "snow_falling", "melted_snow"]
    output_var = file["snow_depth"]
    
    x = file[input_vars]
    y = output_var
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), 
        activation="relu", 
        solver="adam", 
        alpha=0.002, 
        learning_rate_init=0.001, 
        random_state=60, 
        early_stopping=True, 
        max_iter=500
    )
    
    model.fit(x_train_scaled, y_train)
    
    train_score = model.score(x_train_scaled, y_train)
    test_score = model.score(x_test_scaled, y_test)
    
    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"{city_name} Train Score: {train_score:.3f}")
    print(f"{city_name} Test Score: {test_score:.3f}")
    print(f"{city_name} Train RMSE: {train_rmse:.3f}")
    print(f"{city_name} Test RMSE: {test_rmse:.3f}")
    
    with open(out_dir / "snow_depth_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"{city_name} model and scaler saved successfully to {output_dir}!")
    
    return model, scaler, input_vars

def train_all_models():
    """Train models for both Sapporo and Obihiro"""
    
    data_dir = os.path.expanduser("~/Documents/除雪DX/5_snowdepth_app/region_data")
    
    sapporo_csv = os.path.join(data_dir, "sapporo_2016_2025.csv")
    obihiro_csv = os.path.join(data_dir, "obihiro_data_2015.csv")
    
    if not os.path.exists(sapporo_csv):
        print(f"Error: Could not find {sapporo_csv}")
        alt_locations = [
            "~/Documents/除雪DX/5_snowdepth_app/region_data",
            "./",
            "../",
            "~/attachments"
        ]
        
        for loc in alt_locations:
            loc_path = os.path.expanduser(os.path.join(loc, "sapporo_2016_2025.csv"))
            if os.path.exists(loc_path):
                print(f"Found sapporo_2016_2025.csv at {loc_path}")
                sapporo_csv = loc_path
                break
        else:
            print("Could not find sapporo_2016_2025.csv in any location. Please check the file path.")
            return
    
    if not os.path.exists(obihiro_csv):
        print(f"Error: Could not find {obihiro_csv}")
        alt_locations = [
            "~/Documents/除雪DX/5_snowdepth_app/region_data",
            "./",
            "../",
            "~/attachments"
        ]
        
        for loc in alt_locations:
            loc_path = os.path.expanduser(os.path.join(loc, "obihiro_data_2015.csv"))
            if os.path.exists(loc_path):
                print(f"Found obihiro_data_2015.csv at {loc_path}")
                obihiro_csv = loc_path
                break
        else:
            print("Could not find obihiro_data_2015.csv in any location. Please check the file path.")
            return
    
    print(f"Using Sapporo data file: {sapporo_csv}")
    print(f"Using Obihiro data file: {obihiro_csv}")
    
    train_and_save_model(sapporo_csv, "sapporo_models", "Sapporo")
    train_and_save_model(obihiro_csv, "obihiro_models", "Obihiro")
    
    print("\nAll models trained and saved successfully!")

if __name__ == "__main__":
    train_all_models()
