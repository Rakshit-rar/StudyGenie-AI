import streamlit as st
import pandas as pd
import pickle

# --- Configuration --- #
MODEL_PATH = 'model.pkl'
DATA_PATH = 'student_data.csv' # Used to establish correct feature columns for encoding and for categorical options

# List of categorical columns identified during training
CATEGORICAL_COLS = [
    'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
    'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
    'nursery', 'higher', 'internet', 'romantic' # 'romantic' needs to be here
]

# --- Load Model and Reference Data (Cached to avoid reloading on every rerun) ---
@st.cache_resource
def load_resources():
    """Loads the trained model and reference data for feature engineering."""
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)

        # Load original data to get the exact column order after one-hot encoding
        # This is crucial for consistent prediction.
        original_df = pd.read_csv(DATA_PATH)

        # Get unique values for select boxes
        categorical_options = {}
        for col in CATEGORICAL_COLS:
            if col in original_df.columns:
                categorical_options[col] = original_df[col].unique().tolist()

        original_df_encoded = pd.get_dummies(original_df, columns=CATEGORICAL_COLS, drop_first=True)

        # Drop the target variable 'G3' to get the exact feature columns (X)
        reference_feature_columns = original_df_encoded.drop('G3', axis=1).columns

        return model, reference_feature_columns, categorical_options
    except FileNotFoundError:
        st.error(f"Error: Make sure '{MODEL_PATH}' and '{DATA_PATH}' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during resource loading: {e}")
        st.stop()

# --- Preprocessing Function ---
def preprocess_data(new_data: dict, reference_feature_columns):
    """Preprocesses new student data for prediction, aligning with training data."""
    # Convert new data to a pandas DataFrame
    new_df = pd.DataFrame([new_data])

    # Apply one-hot encoding to categorical columns
    processed_df = pd.get_dummies(new_df, columns=CATEGORICAL_COLS, drop_first=True)

    # Align columns with the reference training columns
    final_processed_df = processed_df.reindex(columns=reference_feature_columns, fill_value=0)

    return final_processed_df

# --- Streamlit App Layout ---
def main():
    st.title('StudyGenie AI: Student Performance Predictor')
    st.write('Enter student details to predict their final grade (G3).')

    model, reference_feature_columns, categorical_options = load_resources()

    # --- Input Fields ---
    st.header('Student Information')
    col1, col2, col3 = st.columns(3)

    with col1:
        school = st.selectbox('School', options=categorical_options['school'])
        sex = st.selectbox('Sex', options=categorical_options['sex'])
        age = st.number_input('Age', min_value=15, max_value=22, value=17)
        address = st.selectbox('Address', options=categorical_options['address'])
        famsize = st.selectbox('Family Size', options=categorical_options['famsize'])
        Pstatus = st.selectbox('Parents Cohabitation Status', options=categorical_options['Pstatus'])
        Medu = st.number_input('Mother Education (0-4)', min_value=0, max_value=4, value=2)
        Fedu = st.number_input('Father Education (0-4)', min_value=0, max_value=4, value=2)
        Mjob = st.selectbox('Mother Job', options=categorical_options['Mjob'])
        Fjob = st.selectbox('Father Job', options=categorical_options['Fjob'])
        reason = st.selectbox('Reason to Choose School', options=categorical_options['reason'])
        guardian = st.selectbox('Guardian', options=categorical_options['guardian'])

    with col2:
        traveltime = st.number_input('Travel Time (1-4)', min_value=1, max_value=4, value=2)
        studytime = st.number_input('Study Time (1-4)', min_value=1, max_value=4, value=2)
        failures = st.number_input('Past Class Failures', min_value=0, max_value=4, value=0)
        schoolsup = st.selectbox('Extra Educational Support', options=categorical_options['schoolsup'])
        famsup = st.selectbox('Family Educational Support', options=categorical_options['famsup'])
        paid = st.selectbox('Extra Paid Classes', options=categorical_options['paid'])
        activities = st.selectbox('Extra-curricular Activities', options=categorical_options['activities'])
        nursery = st.selectbox('Attended Nursery School', options=categorical_options['nursery'])
        higher = st.selectbox('Wants to Take Higher Education', options=categorical_options['higher'])

    with col3:
        internet = st.selectbox('Internet Access at Home', options=categorical_options['internet'])
        famrel = st.number_input('Family Relationship Quality (1-5)', min_value=1, max_value=5, value=3)
        freetime = st.number_input('Free Time After School (1-5)', min_value=1, max_value=5, value=3)
        goout = st.number_input('Going Out with Friends (1-5)', min_value=1, max_value=5, value=3)
        health = st.number_input('Current Health Status (1-5)', min_value=1, max_value=5, value=3)
        absences = st.number_input('Number of School Absences', min_value=0, max_value=st.session_state.get('max_absences', 75), value=6)
        G1 = st.number_input('First Period Grade (G1, 0-20)', min_value=0, max_value=20, value=10)
        G2 = st.number_input('Second Period Grade (G2, 0-20)', min_value=0, max_value=20, value=10)

    # Store all inputs in a dictionary
    student_data = {
        'school': school, 'sex': sex, 'age': age, 'address': address, 'famsize': famsize,
        'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu, 'Mjob': Mjob, 'Fjob': Fjob,
        'reason': reason, 'guardian': guardian, 'traveltime': traveltime, 'studytime': studytime,
        'failures': failures, 'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
        'activities': activities, 'nursery': nursery, 'higher': higher, 'internet': internet,
        'famrel': famrel, 'freetime': freetime, 'goout': goout, 'health': health, 
        'absences': absences, 'G1': G1, 'G2': G2
    }
    
    # Set default values for removed features so they don't cause issues for the model
    student_data['romantic'] = 'no'  # Assuming 'no' as default for romantic
    student_data['Dalc'] = 1      # Assuming minimum consumption as default
    student_data['Walc'] = 1      # Assuming minimum consumption as default

    # --- Prediction ---
    if st.button('Predict Final Grade (G3)'):
        if model and reference_feature_columns is not None:
            processed_input = preprocess_data(student_data, reference_feature_columns)
            prediction = model.predict(processed_input)
            st.success(f"Predicted Final Grade (G3): {prediction[0]:.2f}")
        else:
            st.error("Model not loaded. Please ensure model.pkl and student_data.csv are present.")

if __name__ == '__main__':
    main()
