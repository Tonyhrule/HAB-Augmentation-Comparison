import os
import subprocess
import time

def run_command(command):
    print(f"\n=== Running: {command} ===\n")
    start_time = time.time()
    process = subprocess.Popen(command, shell=True)
    process.wait()
    end_time = time.time()
    print(f"\n=== Completed in {end_time - start_time:.2f} seconds ===\n")
    return process.returncode

def main():
    print("Starting Complete Synthetic Data Augmentation Pipeline")
    
    # Step 1: Run basic preprocessing (no synthetic data)
    print("\n=== Step 1: Basic Preprocessing ===")
    if run_command("python preprocess_basic.py") != 0:
        print("Error in basic preprocessing. Exiting.")
        return
    
    # Step 2: Run Gaussian Copula synthetic data generation
    print("\n=== Step 2: Gaussian Copula Synthetic Data Generation ===")
    if run_command("python preprocess_synthetic.py") != 0:
        print("Error in Gaussian Copula synthetic data generation. Exiting.")
        return
    
    # Step 3: Run LLM-based synthetic data generation
    print("\n=== Step 3: LLM Multi-Agent Synthetic Data Generation ===")
    if run_command("python preprocess_llm_synthetic.py") != 0:
        print("Error in LLM Multi-Agent synthetic data generation. Exiting.")
        return
    
    # Step 4: Run comparison training and evaluation
    print("\n=== Step 4: Comparison Training and Evaluation ===")
    if run_command("python train_with_llm.py") != 0:
        print("Error in comparison training and evaluation. Exiting.")
        return
    
    # Step 5: Generate tables for publication
    print("\n=== Step 5: Generate Tables for Publication ===")
    if run_command("python generate_tables.py") != 0:
        print("Error in generating tables. Exiting.")
        return
    
    # Step 6: Generate advanced visualizations
    print("\n=== Step 6: Generate Advanced Visualizations ===")
    if run_command("python generate_visualizations.py") != 0:
        print("Error in generating visualizations. Exiting.")
        return
    
    print("\n=== Pipeline Completed Successfully ===")
    print("Results are available in the following directories:")
    print("- Processed data: output/")
    print("- Trained models: models/")
    print("- Performance tables: tables/")
    print("- Visualizations: figures/")
    
    # Verify output files
    print("\n=== Verifying Output Files ===")
    verify_files()

def verify_files():
    """Verify that all expected output files have been created"""
    expected_files = [
        "output/processed_data.pkl",
        "output/processed_data_with_synthetic.pkl",
        "output/processed_data_with_llm_synthetic.pkl",
        "output/cost_gaussian_copula.json",
        "output/cost_llm_multi_agent.json",
        "models/model_non_synthetic.pkl",
        "models/model_with_gaussian_synthetic.pkl",
        "models/model_with_llm_synthetic.pkl",
        "figures/synthetic_methods_comparison.png",
        "figures/radar_chart.png",
        "figures/correlation_heatmaps.png",
        "figures/error_violin_plot.png",
        "figures/parallel_coordinates.png",
        "figures/combined_visualization.png",
        "tables/performance_metrics.csv",
        "tables/performance_metrics.tex",
        "tables/performance_metrics.md",
        "tables/computational_cost.csv",
        "tables/computational_cost.tex",
        "tables/computational_cost.md",
        "tables/detailed_metrics.csv",
        "tables/detailed_metrics.tex",
        "tables/detailed_metrics.md",
        "tables/feature_importance.csv",
        "tables/feature_importance.tex",
        "tables/feature_importance.md"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("\nWarning: The following expected files are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
    else:
        print("\nAll expected output files have been created successfully.")

if __name__ == "__main__":
    main()
