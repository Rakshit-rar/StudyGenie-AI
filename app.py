import pandas as pd
import pickle

# --- Configuration --- #
MODEL_PATH = 'model.pkl'
DATA_PATH = 'student_data.csv' # Used to establish correct feature columns for encoding

# List of categorical columns identified during training
CATEGORICAL_COLS = [
    'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
    'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 
    'nursery', 'higher', 'internet', 'romantic'
]

# --- Load Model and Reference Data --- #
def load_resources():
    """Loads the trained model and reference data for feature engineering."""
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded successfully from {MODEL_PATH}")

        # Load original data to get the exact column order after one-hot encoding
        # This is crucial for consistent prediction.
        original_df = pd.read_csv(DATA_PATH)
        original_df_encoded = pd.get_dummies(original_df, columns=CATEGORICAL_COLS, drop_first=True)
        
        # Drop the target variable 'G3' to get the exact feature columns (X)
        reference_feature_columns = original_df_encoded.drop('G3', axis=1).columns
        print(f"Reference feature columns loaded from {DATA_PATH}")

        return model, reference_feature_columns
    except FileNotFoundError:
        print(f"Error: Make sure '{MODEL_PATH}' and '{DATA_PATH}' are in the same directory.")
        return None, None
    except Exception as e:
        print(f"An error occurred during resource loading: {e}")
        return None, None

# --- Preprocessing Function --- #
def preprocess_data(new_data: dict, reference_feature_columns):
    """Preprocesses new student data for prediction."""
    # Convert new data to a pandas DataFrame
    new_df = pd.DataFrame([new_data])

    # Apply one-hot encoding to categorical columns
    # Use 'reindex' to ensure new_df has the same columns as the training data, 
    # filling new columns with 0 if they don't appear in the new data.
    processed_df = pd.get_dummies(new_df, columns=CATEGORICAL_COLS, drop_first=True)
    
    # Align columns with the reference training columns
    # Fill any missing columns (due to categories not present in new_df) with 0
    # and drop any extra columns not present in the original training set.
    final_processed_df = processed_df.reindex(columns=reference_feature_columns, fill_value=0)

    return final_processed_df

# --- Prediction Function --- #
def predict_performance(student_data: dict, model, reference_feature_columns):
    """Predicts the final grade (G3) for a given student."""
    if model is None or reference_feature_columns is None:
        print("Model or reference features not loaded. Cannot make prediction.")
        return None

    processed_input = preprocess_data(student_data, reference_feature_columns)
    prediction = model.predict(processed_input)
    return prediction[0]

# --- Main Execution Block --- #
if __name__ == "__main__":
    # Load model and reference data once at startup
    model, reference_feature_columns = load_resources()

    if model and reference_feature_columns is not None:
        print("\nReady to make predictions!")

        # Example student data (ensure all relevant columns are present)
        # You can replace this with input from a web form or API request
        example_student_data = {
            'school': 'GP', 'sex': 'F', 'age': 18, 'address': 'U', 'famsize': 'GT3', 
            'Pstatus': 'A', 'Medu': 4, 'Fedu': 4, 'Mjob': 'at_home', 'Fjob': 'teacher', 
            'reason': 'course', 'guardian': 'mother', 'traveltime': 2, 'studytime': 2, 
            'failures': 0, 'schoolsup': 'yes', 'famsup': 'no', 'paid': 'no', 
            'activities': 'no', 'nursery': 'yes', 'higher': 'yes', 'internet': 'no', 
            'romantic': 'no', 'famrel': 4, 'freetime': 3, 'goout': 4, 'Dalc': 1, 
            'Walc': 1, 'health': 3, 'absences': 6, 'G1': 5, 'G2': 6
        }

        predicted_g3 = predict_performance(example_student_data, model, reference_feature_columns)

        if predicted_g3 is not None:
            print(f"\nPredicted G3 for the example student: {predicted_g3:.2f}")

        # Another example with slightly different values
        example_student_data_2 = {
            'school': 'MS', 'sex': 'M', 'age': 17, 'address': 'R', 'famsize': 'LE3', 
            'Pstatus': 'T', 'Medu': 2, 'Fedu': 2, 'Mjob': 'other', 'Fjob': 'other', 
            'reason': 'reputation', 'guardian': 'father', 'traveltime': 1, 'studytime': 3, 
            'failures': 0, 'schoolsup': 'no', 'famsup': 'yes', 'paid': 'yes', 
            'activities': 'yes', 'nursery': 'yes', 'higher': 'yes', 'internet': 'yes', 
            'romantic': 'yes', 'famrel': 5, 'freetime': 5, 'goout': 5, 'Dalc': 2, 
            'Walc': 3, 'health': 2, 'absences': 0, 'G1': 10, 'G2': 11
        }

        predicted_g3_2 = predict_performance(example_student_data_2, model, reference_feature_columns)

        if predicted_g3_2 is not None:
            print(f"Predicted G3 for the second example student: {predicted_g3_2:.2f}")
