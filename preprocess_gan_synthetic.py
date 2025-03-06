import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv
from ctgan import CTGAN

# Load environment variables
load_dotenv()

# Constants
DATA_PATH = "Dataset.xlsx"
OUTPUT_DIR = "output"
SYNTHETIC_DATA_ROWS = 250
TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURES = ["Temperature", "Salinity", "UVB"]
TARGET = "ChlorophyllaFlor"

def load_data(data_path):
    """Load data from Excel file"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    return pd.read_excel(data_path)

def validate_structure(df):
    """Validate that the DataFrame contains the required columns"""
    required_columns = FEATURES + [TARGET]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame does not contain required columns: {required_columns}")
    print("Dataset structure validated.")

def clean_data(data):
    """Remove rows with missing values"""
    original_shape = data.shape
    data = data.dropna()
    print(f"Original shape: {original_shape}, Cleaned shape: {data.shape}")
    return data

def transform_target(data):
    """Apply log transformation to the target variable"""
    print("Applying log transformation to the target variable...")
    data[TARGET] = np.log1p(data[TARGET])
    return data

def generate_ctgan_synthetic_data(data, num_samples):
    """Generate synthetic data using CTGAN"""
    print("Generating synthetic data using CTGAN...")
    
    # Track computational cost
    from utils.cost_tracker import CostTracker
    cost_tracker = CostTracker("ctgan").start()
    
    # Define CTGAN model with appropriate hyperparameters
    ctgan = CTGAN(
        epochs=300,
        batch_size=500,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        embedding_dim=128,
        verbose=True,
        cuda=False  # Set to True if GPU is available
    )
    
    # Fit the model on the data
    print("Fitting CTGAN model on the data...")
    ctgan.fit(data)
    
    # Generate synthetic samples
    print(f"Generating {num_samples} synthetic samples...")
    synthetic_data = ctgan.sample(num_samples)
    
    # Clip synthetic data to real data range
    synthetic_data = clip_synthetic_data(synthetic_data, data)
    
    # Stop tracking and save metrics
    cost_tracker.stop()
    cost_tracker.save_to_file("output/cost_ctgan.json")
    print(f"Execution time: {cost_tracker.get_execution_time():.2f} seconds")
    print("Synthetic data generated successfully.")
    
    return synthetic_data

def clip_synthetic_data(synthetic_data, real_data):
    """Clip synthetic data to the range of real data"""
    for column in synthetic_data.columns:
        synthetic_data[column] = synthetic_data[column].clip(
            real_data[column].min(), 
            real_data[column].max()
        )
    return synthetic_data

def perform_ks_test(data, synthetic_data):
    """Perform Kolmogorov-Smirnov test to compare distributions"""
    print("\nPerforming Kolmogorov-Smirnov test to validate synthetic data...")
    for column in FEATURES + [TARGET]:
        stat, p_value = ks_2samp(data[column], synthetic_data[column])
        print(f"{column}: KS Statistic = {stat:.4f}, p-value = {p_value:.4f}")
        if p_value < 0.05:
            print(f"  Warning: Distributions for {column} are significantly different (p < 0.05)")
        else:
            print(f"  Distributions for {column} are not significantly different (p >= 0.05)")

def preprocess_and_combine(real_data, synthetic_data, weight=0.20):
    """Preprocess and combine real and synthetic data"""
    print("\nPreprocessing and combining real and synthetic data...")
    
    # Sample a fraction of synthetic data based on weight parameter
    synthetic_data_sampled = synthetic_data.sample(
        frac=weight, 
        random_state=RANDOM_STATE, 
        replace=True
    )
    combined_data = pd.concat([real_data, synthetic_data_sampled], ignore_index=True)
    
    print(f"Real data shape: {real_data.shape}")
    print(f"Synthetic data sampled shape: {synthetic_data_sampled.shape}")
    print(f"Combined data shape: {combined_data.shape}")

    X = combined_data[FEATURES]
    y = combined_data[TARGET]

    # Generate polynomial features for interaction terms
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    print(f"Polynomial features shape: {X_poly.shape}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing set shape: X={X_test.shape}, y={y_test.shape}")

    # Use median imputation for robustness to outliers
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def reverse_log_transformation(y_test, y_pred):
    """Reverse the log transformation applied earlier"""
    y_test_reversed = np.expm1(y_test)
    y_pred_reversed = np.expm1(y_pred)
    return y_test_reversed, y_pred_reversed

def evaluate_model_performance(y_test, y_pred):
    """Evaluate model performance using standard regression metrics"""
    # Convert predictions back to original scale before calculating metrics
    y_test_reversed, y_pred_reversed = reverse_log_transformation(y_test, y_pred)
    
    # Calculate standard regression metrics
    mse = mean_squared_error(y_test_reversed, y_pred_reversed)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_reversed, y_pred_reversed)
    r2 = r2_score(y_test_reversed, y_pred_reversed)
    
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    }

def train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test):
    """Train and evaluate Ridge Regression and Random Forest models"""
    print("\nTraining and evaluating models...")
    
    # Train and evaluate Ridge Regression model
    print("Ridge Regression Performance:")
    ridge_model = Ridge(alpha=10.0)
    ridge_model.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_model.predict(X_test_scaled)
    ridge_metrics = evaluate_model_performance(y_test, y_pred_ridge)
    
    # Train and evaluate Random Forest model with optimized hyperparameters
    print("\nRandom Forest Performance:")
    rf_model = RandomForestRegressor(
        n_estimators=600, 
        max_depth=30, 
        min_samples_split=4, 
        random_state=RANDOM_STATE
    )
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    rf_metrics = evaluate_model_performance(y_test, y_pred_rf)
    
    # Save the best model (assuming Random Forest is better)
    print("\nSaving the Random Forest model...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, os.path.join("models", "model_with_gan_synthetic.pkl"))
    
    return rf_metrics

def save_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler, output_dir):
    """Save processed data and scaler for later use"""
    print(f"\nSaving processed data and scaler to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    joblib.dump(
        (X_train_scaled, X_test_scaled, y_train, y_test), 
        os.path.join(output_dir, "processed_data_with_gan_synthetic.pkl")
    )
    
    # Save scaler
    joblib.dump(
        scaler, 
        os.path.join(output_dir, "scaler_with_gan_synthetic.pkl")
    )
    
    print("Data and scaler saved successfully.")

def main():
    """Main function to run the entire pipeline"""
    print("Starting CTGAN synthetic data generation pipeline...")
    
    # Load and preprocess data
    data = load_data(DATA_PATH)
    validate_structure(data)
    data_cleaned = clean_data(data)
    data_transformed = transform_target(data_cleaned)
    
    # Generate synthetic data
    synthetic_data = generate_ctgan_synthetic_data(data_transformed, SYNTHETIC_DATA_ROWS)
    
    # Validate synthetic data
    perform_ks_test(data_transformed, synthetic_data)
    
    # Preprocess and combine data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_and_combine(
        data_transformed, synthetic_data, weight=0.20
    )
    
    # Train and evaluate models
    metrics = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Save data and scaler
    save_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler, OUTPUT_DIR)
    
    print("\nCTGAN synthetic data generation pipeline completed successfully.")
    return metrics

if __name__ == "__main__":
    main()
