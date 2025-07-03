import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter the passenger details to predict survival:")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 6, 0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Prepare input for prediction
input_data = {
    'pclass': [pclass],
    'age': [age],
    'sibsp': [sibsp],
    'parch': [parch],
    'sex_male': [1 if sex == 'male' else 0],
    'embarked_Q': [1 if embarked == 'Q' else 0],
    'embarked_S': [1 if embarked == 'S' else 0]
}

input_df = pd.DataFrame(input_data)

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("🎉 This passenger would have SURVIVED!")
    else:
        st.error("💀 This passenger would NOT have survived.")
