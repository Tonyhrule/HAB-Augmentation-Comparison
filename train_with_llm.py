import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH_WITH_GAUSSIAN = "output/processed_data_with_synthetic.pkl"
DATA_PATH_WITH_LLM = "output/processed_data_with_llm_synthetic.pkl"
DATA_PATH_NON_SYNTHETIC = "output/processed_data.pkl"
MODEL_OUTPUT_DIR = "models"
FIGURES_DIR = "figures"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_data(path):
    data = joblib.load(path)
    print(f"Loaded data shapes: X_train={data[0].shape}, y_train={data[2].shape}, X_test={data[1].shape}, y_test={data[3].shape}")
    return data

def reshape_target(y):
    # Ensure target variable is 1D for compatibility with sklearn estimators
    if len(y.shape) > 1:
        y = y.ravel()
    return y

def evaluate_model(model, X_test, y_test):
    # Calculate standard regression metrics for model evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}

def train_and_evaluate(X_train, X_test, y_train, y_test, output_model_path, use_grid_search=False):
    # Ensure target variables are in the correct shape
    y_train = reshape_target(y_train)
    y_test = reshape_target(y_test)

    # Initialize gradient boosting regressor
    gb = GradientBoostingRegressor(random_state=42)

    # Perform hyperparameter tuning if requested
    if use_grid_search:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        }
        grid_search = GridSearchCV(
            gb, param_grid, scoring="r2", cv=5, n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        gb = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        # Train with default parameters
        gb.fit(X_train, y_train)

    # Evaluate model performance
    metrics = evaluate_model(gb, X_test, y_test)
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save trained model
    joblib.dump(gb, output_model_path)

    return metrics

def plot_comparison(non_synthetic_metrics, gaussian_metrics, llm_metrics):
    """
    Create a 2x2 grid of bar charts comparing performance metrics across three models.
    Highlights the best performing model for each metric with a distinct color.
    """
    metrics = ["MSE", "RMSE", "MAE", "R²"]
    models = ["Non-Synthetic", "Gaussian Copula", "LLM Multi-Agent"]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Define colors for visual distinction
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    
    for i, metric in enumerate(metrics):
        values = [
            non_synthetic_metrics[metric],
            gaussian_metrics[metric],
            llm_metrics[metric]
        ]
        
        # For R², higher is better, so we invert the comparison
        if metric == "R²":
            best_idx = values.index(max(values))
        else:
            best_idx = values.index(min(values))
        
        # Highlight best model with color, others in light gray
        bar_colors = [colors[j] if j == best_idx else "lightgray" for j in range(len(models))]
        
        axes[i].bar(models, values, color=bar_colors)
        axes[i].set_title(f"{metric} Comparison", fontsize=14, weight="bold")
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].grid(axis="y", alpha=0.3)
        
        # Add value labels on top of bars for better readability
        for j, v in enumerate(values):
            axes[i].text(j, v * 1.05, f"{v:.4f}", ha="center", fontsize=10, weight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "synthetic_methods_comparison.png"), dpi=300)
    plt.close()

def main():
    print("Training with non-synthetic data...")
    X_train_non, X_test_non, y_train_non, y_test_non = load_data(DATA_PATH_NON_SYNTHETIC)[:4]
    non_synthetic_metrics = train_and_evaluate(
        X_train_non, X_test_non, y_train_non, y_test_non,
        os.path.join(MODEL_OUTPUT_DIR, "model_non_synthetic.pkl"),
        use_grid_search=True
    )

    print("\nTraining with Gaussian Copula synthetic data...")
    X_train_gaussian, X_test_gaussian, y_train_gaussian, y_test_gaussian = load_data(DATA_PATH_WITH_GAUSSIAN)[:4]
    gaussian_metrics = train_and_evaluate(
        X_train_gaussian, X_test_gaussian, y_train_gaussian, y_test_gaussian,
        os.path.join(MODEL_OUTPUT_DIR, "model_with_gaussian_synthetic.pkl"),
        use_grid_search=True
    )

    print("\nTraining with LLM Multi-Agent synthetic data...")
    X_train_llm, X_test_llm, y_train_llm, y_test_llm = load_data(DATA_PATH_WITH_LLM)[:4]
    llm_metrics = train_and_evaluate(
        X_train_llm, X_test_llm, y_train_llm, y_test_llm,
        os.path.join(MODEL_OUTPUT_DIR, "model_with_llm_synthetic.pkl"),
        use_grid_search=True
    )

    print("\n--- Final Summary ---")
    print("Metrics for Non-Synthetic Data:")
    for key, value in non_synthetic_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nMetrics for Gaussian Copula Synthetic Data:")
    for key, value in gaussian_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nMetrics for LLM Multi-Agent Synthetic Data:")
    for key, value in llm_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Plot comparison
    plot_comparison(non_synthetic_metrics, gaussian_metrics, llm_metrics)
    print(f"\nComparison plot saved to {os.path.join(FIGURES_DIR, 'synthetic_methods_comparison.png')}")

if __name__ == "__main__":
    main()
