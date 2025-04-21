import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('logistic_model.pkl', 'rb'))

st.title('Titanic Survival Prediction')

Pclass = st.selectbox('Passenger Class', [1, 2, 3])
Sex_male = st.radio('Sex', ['Male', 'Female']) == 'Male'
Age = st.slider('Age', 0, 80, 25)
SibSp = st.number_input('Siblings/Spouse Aboard', 0, 10, 0)
Parch = st.number_input('Parents/Children Aboard', 0, 10, 0)
Fare = st.number_input('Fare', 0.0, 600.0, 50.0)
Embarked_Q = st.checkbox('Embarked Q')
Embarked_S = st.checkbox('Embarked S')

input_data = np.array([[Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q, Embarked_S]])
if st.button('Predict'):
    result = model.predict(input_data)
    st.success('Survived' if result[0] else 'Did not survive')
