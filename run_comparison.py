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
    print("Starting Synthetic Data Augmentation Comparison Pipeline")
    
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
    
    print("\n=== Pipeline Completed Successfully ===")
    print("Results are available in the 'figures' directory.")
    print("Synthetic data comparison plot: figures/synthetic_methods_comparison.png")

if __name__ == "__main__":
    main()
