import pandas as pd
import pickle as pk
import streamlit as st
import os

# ==============================
# Load Model
# ==============================
if os.path.exists('bike_model.pkl'):
    model = pk.load(open('bike_model.pkl', 'rb'))
else:
    st.error("bike_model.pkl file not found!")
    st.stop()

# ==============================
# Load Dataset
# ==============================
bikes_data = pd.read_csv('Used_Bikes.csv')

# For encoding
brand_mapping = {b: i+1 for i, b in enumerate(bikes_data['brand'].unique())}
city_mapping = {c: i+1 for i, c in enumerate(bikes_data['city'].unique())}
owner_mapping = {
    'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth Owner Or More': 4
}

# ==============================
# Streamlit UI
# ==============================
st.title("🏍️ Used Bike Price Prediction")
st.write("Fill in the details below to estimate your bike's selling price.")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox('Bike Brand', sorted(bikes_data['brand'].unique()))
    city = st.selectbox('City', sorted(bikes_data['city'].unique()))
    owner = st.selectbox('Owner Type', list(owner_mapping.keys()))
with col2:
    kms_driven = st.number_input('Kms Driven', min_value=0, max_value=300000, value=10000)
    age = st.number_input('Age (years)', min_value=0, max_value=50, value=5)
    power = st.number_input('Engine Power (cc)', min_value=50, max_value=2000, value=150)

# ==============================
# Prediction
# ==============================
if st.button("🔍 Predict Price"):
    # Prepare input in the same order as training
    feature_columns = ['brand', 'city', 'kms_driven', 'owner', 'age', 'power']
    try:
        input_data = pd.DataFrame(
            [[brand_mapping[brand], city_mapping[city], kms_driven, owner_mapping[owner], age, power]],
            columns=feature_columns
        )
        price = model.predict(input_data)[0]
        st.success(f"💰 Estimated Bike Price: ₹ {price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")