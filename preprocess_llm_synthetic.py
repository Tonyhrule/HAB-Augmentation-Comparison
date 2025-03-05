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
import openai
import json
import time

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
DATA_PATH = "Dataset.xlsx"
OUTPUT_DIR = "output"
SYNTHETIC_DATA_ROWS = 250
TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURES = ["Temperature", "Salinity", "UVB"]
TARGET = "ChlorophyllaFlor"

def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    return pd.read_excel(data_path)

def validate_structure(df):
    required_columns = FEATURES + [TARGET]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame does not contain required columns: {required_columns}")
    print("Dataset structure validated.")

def clean_data(data):
    original_shape = data.shape
    data = data.dropna()
    print(f"Original shape: {original_shape}, Cleaned shape: {data.shape}")
    return data

def transform_target(data):
    print("Applying log transformation to the target variable...")
    data[TARGET] = np.log1p(data[TARGET])
    return data

def get_data_statistics(data):
    """Get statistics for each feature to guide the LLM agents"""
    stats = {}
    for column in data.columns:
        stats[column] = {
            "min": float(data[column].min()),
            "max": float(data[column].max()),
            "mean": float(data[column].mean()),
            "median": float(data[column].median()),
            "std": float(data[column].std())
        }
    return stats

def get_feature_correlations(data):
    """Get correlations between features to guide the LLM agents"""
    return data.corr().to_dict()

def create_system_prompt():
    """Create the system prompt for the data generation agent"""
    return """You are a specialized data generation agent for harmful algal bloom detection. 
Your task is to generate realistic synthetic data that maintains the statistical properties 
and relationships of the original dataset. You will be provided with statistics and correlations 
from the original dataset to guide your generation process. The generated data should be realistic, 
diverse, and maintain the relationships between features."""

def create_domain_expert_prompt(stats, correlations):
    """Create the prompt for the domain expert agent"""
    return f"""As a domain expert in harmful algal blooms and oceanography, review and validate 
the synthetic data to ensure it aligns with domain knowledge. Consider the following:

1. Temperature and Salinity typically have an inverse relationship in coastal waters.
2. UVB radiation varies with season and latitude.
3. Chlorophyll-a fluorescence (algal biomass) is influenced by temperature, salinity, and UVB.

Dataset statistics: {json.dumps(stats, indent=2)}

Feature correlations: {json.dumps(correlations, indent=2)}

Provide feedback on the synthetic data and suggest adjustments if necessary."""

def create_data_scientist_prompt(stats, correlations):
    """Create the prompt for the data scientist agent"""
    return f"""As a data scientist, analyze the statistical properties of the synthetic data 
and compare them with the original dataset. Consider the following:

1. The distribution of each feature should be similar to the original.
2. The correlations between features should be preserved.
3. The range of values should be realistic and within the bounds of the original data.

Original dataset statistics: {json.dumps(stats, indent=2)}

Original feature correlations: {json.dumps(correlations, indent=2)}

Provide feedback on the synthetic data and suggest adjustments to improve its quality."""

def generate_synthetic_sample(data_stats, correlations, sample_index, cost_tracker=None):
    """Generate a single synthetic sample using the LLM"""
    # Create a prompt for generating a single data point
    prompt = f"""Generate a single realistic data point for harmful algal bloom detection.
The data should have the following features: {FEATURES} and target: {TARGET}.

Use these statistics to guide your generation:
{json.dumps(data_stats, indent=2)}

And maintain these feature correlations:
{json.dumps(correlations, indent=2)}

This is sample {sample_index} of {SYNTHETIC_DATA_ROWS}.

Return ONLY a JSON object with the feature values and target, no explanation or other text.
Format: {{"Temperature": value, "Salinity": value, "UVB": value, "ChlorophyllaFlor": value}}"""
    
    # Call the OpenAI API
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": create_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        # Track API call and tokens if cost_tracker is provided
        if cost_tracker:
            cost_tracker.record_api_call(tokens_used=response.usage.total_tokens)
        
        # Extract and parse the generated data point
        content = response.choices[0].message.content.strip()
        # Handle potential JSON formatting issues
        try:
            # Try to parse as is
            data_point = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    data_point = json.loads(json_match.group(0))
                except:
                    # If all else fails, generate a random point based on statistics
                    data_point = {
                        feature: np.random.normal(
                            data_stats[feature]["mean"], 
                            data_stats[feature]["std"]
                        ) for feature in FEATURES + [TARGET]
                    }
            else:
                # If no JSON-like structure found, generate random point
                data_point = {
                    feature: np.random.normal(
                        data_stats[feature]["mean"], 
                        data_stats[feature]["std"]
                    ) for feature in FEATURES + [TARGET]
                }
        
        return data_point
    except Exception as e:
        print(f"Error generating synthetic sample {sample_index}: {e}")
        # Fallback to random generation based on statistics
        return {
            feature: np.random.normal(
                data_stats[feature]["mean"], 
                data_stats[feature]["std"]
            ) for feature in FEATURES + [TARGET]
        }

def validate_synthetic_sample(sample, data_stats, domain_expert_feedback=None, cost_tracker=None):
    """Validate a synthetic sample using the domain expert agent"""
    if domain_expert_feedback is None:
        domain_expert_feedback = ""
    
    prompt = f"""Validate this synthetic data point for harmful algal bloom detection:
{json.dumps(sample, indent=2)}

Original data statistics:
{json.dumps(data_stats, indent=2)}

Previous feedback: {domain_expert_feedback}

Is this data point realistic and valid? If not, what adjustments are needed?
Return your response as a JSON object with 'valid' (boolean), 'adjustments' (object with feature names and new values if needed), and 'feedback' (string with general feedback for future samples).
Format: {{"valid": true/false, "adjustments": {{"feature": value}}, "feedback": "Your feedback here"}}"""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": create_domain_expert_prompt(data_stats, {})},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        # Track API call and tokens if cost_tracker is provided
        if cost_tracker:
            cost_tracker.record_api_call(tokens_used=response.usage.total_tokens)
        
        content = response.choices[0].message.content.strip()
        try:
            validation = json.loads(content)
            if not validation.get("valid", True):
                # Apply adjustments
                for feature, value in validation.get("adjustments", {}).items():
                    if feature in sample:
                        sample[feature] = value
            
            # Add feedback to the sample for iterative improvement
            if "feedback" in validation:
                sample["feedback"] = validation["feedback"]
        except json.JSONDecodeError:
            # If parsing fails, assume the sample is valid
            print("Error parsing domain expert feedback. Assuming sample is valid.")
        
        return sample
    except Exception as e:
        print(f"Error validating synthetic sample: {e}")
        return sample

def refine_synthetic_dataset(synthetic_data, original_data_stats, correlations, cost_tracker=None):
    """Refine the synthetic dataset using the data scientist agent"""
    # Calculate statistics of the synthetic data
    synthetic_stats = get_data_statistics(synthetic_data)
    
    # Create a prompt for the data scientist agent
    prompt = f"""Analyze and refine this synthetic dataset for harmful algal bloom detection.
    
Original data statistics:
{json.dumps(original_data_stats, indent=2)}

Synthetic data statistics:
{json.dumps(synthetic_stats, indent=2)}

Original correlations:
{json.dumps(correlations, indent=2)}

Synthetic correlations:
{json.dumps(synthetic_data.corr().to_dict(), indent=2)}

What adjustments should be made to make the synthetic data more realistic and statistically similar to the original?
Provide specific recommendations for scaling or transforming features.
Return your response as a JSON object with 'adjustments' containing transformation instructions.
Format: {{"adjustments": {{"feature": {{"operation": "scale/shift/transform", "parameters": {{"param1": value}}}}}}}}"""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": create_data_scientist_prompt(original_data_stats, correlations)},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        # Track API call and tokens if cost_tracker is provided
        if cost_tracker:
            cost_tracker.record_api_call(tokens_used=response.usage.total_tokens)
        
        content = response.choices[0].message.content.strip()
        try:
            # Parse the feedback
            feedback = json.loads(content)
            adjustments = feedback.get("adjustments", {})
            
            # Apply the adjustments
            for feature, adjustment in adjustments.items():
                if feature in synthetic_data.columns:
                    operation = adjustment.get("operation", "").lower()
                    parameters = adjustment.get("parameters", {})
                    
                    if operation == "scale":
                        factor = parameters.get("factor", 1.0)
                        synthetic_data[feature] = synthetic_data[feature] * factor
                    elif operation == "shift":
                        offset = parameters.get("offset", 0.0)
                        synthetic_data[feature] = synthetic_data[feature] + offset
                    elif operation == "transform":
                        transform_type = parameters.get("type", "")
                        if transform_type == "log":
                            synthetic_data[feature] = np.log1p(synthetic_data[feature])
                        elif transform_type == "exp":
                            synthetic_data[feature] = np.expm1(synthetic_data[feature])
                        elif transform_type == "sqrt":
                            synthetic_data[feature] = np.sqrt(synthetic_data[feature])
            
            # Ensure values are within original range
            for column in synthetic_data.columns:
                min_val = original_data_stats[column]["min"]
                max_val = original_data_stats[column]["max"]
                synthetic_data[column] = synthetic_data[column].clip(min_val, max_val)
                
        except json.JSONDecodeError:
            print("Error parsing data scientist feedback. Applying basic refinements.")
            # Fallback to basic refinements
            for column in synthetic_data.columns:
                min_val = original_data_stats[column]["min"]
                max_val = original_data_stats[column]["max"]
                synthetic_data[column] = synthetic_data[column].clip(min_val, max_val)
        
        return synthetic_data
    except Exception as e:
        print(f"Error refining synthetic dataset: {e}")
        # Fallback to basic refinements
        for column in synthetic_data.columns:
            min_val = original_data_stats[column]["min"]
            max_val = original_data_stats[column]["max"]
            synthetic_data[column] = synthetic_data[column].clip(min_val, max_val)
        return synthetic_data

def generate_llm_synthetic_data(data, num_samples):
    """Generate synthetic data using the LLM collaborative multi-agent pipeline"""
    print("Generating synthetic data using LLM collaborative multi-agent pipeline...")
    
    # Track computational cost
    from utils.cost_tracker import CostTracker
    cost_tracker = CostTracker("llm_multi_agent").start()
    
    # Get statistics and correlations from the original data
    data_stats = get_data_statistics(data)
    correlations = get_feature_correlations(data)
    
    # Generate synthetic samples with iterative feedback
    synthetic_samples = []
    domain_expert_feedback = None
    batch_size = 10  # Process in batches for efficiency
    
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_samples = []
        
        print(f"Generating samples {batch_start+1}-{batch_end}/{num_samples}...")
        
        # Generate batch of samples
        for i in range(batch_start, batch_end):
            # Generate a synthetic sample
            sample = generate_synthetic_sample(data_stats, correlations, i+1, cost_tracker)
            
            # Validate all samples with the domain expert agent
            sample = validate_synthetic_sample(sample, data_stats, domain_expert_feedback, cost_tracker)
            # Store feedback for iterative improvement
            domain_expert_feedback = sample.get("feedback", domain_expert_feedback)
            
            # Remove feedback from sample before adding to batch
            if "feedback" in sample:
                del sample["feedback"]
                
            batch_samples.append(sample)
            synthetic_samples.append(sample)
            
            # Add a small delay to avoid rate limiting
            if (i - batch_start) % 5 == 0:
                time.sleep(0.5)
        
        # Convert batch to DataFrame for data scientist refinement
        if batch_samples:
            batch_df = pd.DataFrame(batch_samples)
            
            # Get data scientist feedback on the batch
            batch_df = refine_synthetic_dataset(batch_df, data_stats, correlations, cost_tracker)
            
            # Update the synthetic samples with refined values
            for j, sample in enumerate(batch_samples):
                for column in sample.keys():
                    if column in batch_df.columns:
                        synthetic_samples[batch_start + j][column] = batch_df.iloc[j][column]
    
    # Convert all samples to DataFrame
    synthetic_data = pd.DataFrame(synthetic_samples)
    
    # Final refinement of the complete dataset
    synthetic_data = refine_synthetic_dataset(synthetic_data, data_stats, correlations, cost_tracker)
    
    # Stop tracking and save metrics
    cost_tracker.stop()
    cost_tracker.save_to_file("output/cost_llm_multi_agent.json")
    print(f"Execution time: {cost_tracker.get_execution_time():.2f} seconds")
    print(f"API calls: {cost_tracker.api_calls}, Total tokens: {cost_tracker.api_tokens}")
    print(f"Estimated API cost: ${cost_tracker.get_api_cost():.4f}")
    
    print("Synthetic data generated successfully.")
    return synthetic_data

def clip_synthetic_data(synthetic_data, real_data):
    for column in synthetic_data.columns:
        synthetic_data[column] = synthetic_data[column].clip(real_data[column].min(), real_data[column].max())
    return synthetic_data

def perform_ks_test(data, synthetic_data):
    print("\nPerforming KS test...")
    for column in FEATURES + [TARGET]:
        stat, p_value = ks_2samp(data[column], synthetic_data[column])
        print(f"{column}: KS Statistic = {stat}, p-value = {p_value}")

def preprocess_and_combine(real_data, synthetic_data, weight=0.20):
    print("Combining real and synthetic data...")
    synthetic_data_sampled = synthetic_data.sample(frac=weight, random_state=RANDOM_STATE, replace=True)
    combined_data = pd.concat([real_data, synthetic_data_sampled], ignore_index=True)

    X = combined_data[FEATURES]
    y = combined_data[TARGET]

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def reverse_log_transformation(y_test, y_pred):
    y_test_reversed = np.expm1(y_test)
    y_pred_reversed = np.expm1(y_pred)
    return y_test_reversed, y_pred_reversed

def evaluate_model_performance(y_test, y_pred):
    y_test_reversed, y_pred_reversed = reverse_log_transformation(y_test, y_pred)
    mse = mean_squared_error(y_test_reversed, y_pred_reversed)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_reversed, y_pred_reversed)
    r2 = r2_score(y_test_reversed, y_pred_reversed)
    print(f"  MSE: {mse}")
    print(f"  RMSE: {rmse}")
    print(f"  MAE: {mae}")
    print(f"  R² Score: {r2}")

def train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test):
    print("\nTraining Ridge Regression model...")
    ridge_model = Ridge(alpha=10.0)
    ridge_model.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_model.predict(X_test_scaled)
    print("Ridge Regression Performance:")
    evaluate_model_performance(y_test, y_pred_ridge)

    print("\nTraining Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=600, max_depth=30, min_samples_split=4, random_state=RANDOM_STATE)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    print("Random Forest Performance:")
    evaluate_model_performance(y_test, y_pred_rf)

def save_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), os.path.join(output_dir, "processed_data_with_llm_synthetic.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler_with_llm_synthetic.pkl"))

def main():
    data = load_data(DATA_PATH)
    validate_structure(data)

    data_cleaned = clean_data(data)
    data_transformed = transform_target(data_cleaned)

    synthetic_data = generate_llm_synthetic_data(data_transformed, SYNTHETIC_DATA_ROWS)

    perform_ks_test(data_transformed, synthetic_data)

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_and_combine(data_transformed, synthetic_data, weight=0.20)

    train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)

    save_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler, OUTPUT_DIR)

if __name__ == "__main__":
    main()
