import streamlit as st
import joblib
import pandas as pd
import os

# Get the exact folder where this Python file is located (the CODE folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "gradient_boosting_model.pkl")

# Load the model safely using its absolute path
model = joblib.load(model_path)


st.title("🏡 House Price Prediction App")
st.markdown("Please provide the details of the property below, and the AI will estimate its market value.")

st.divider()

# Create a form to wrap all input elements cleanly
with st.form("house_prediction_form"):
    
    # Section 1: General & Building Classification
    st.subheader("1. General & Building Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mssubclass = st.selectbox(
            "Building Style / Type", 
            [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190],
            help="Type of dwelling (e.g., 20 = 1-Story 1946+, 60 = 2-Story 1946+)"
        )
        lot_area = st.number_input("Total Land Area (sq ft)", min_value=0, value=10000, step=500, help="Size of the property/plot in square feet.")
        
    with col2:
        overall_qual = st.slider("Material & Finish Quality", min_value=1, max_value=10, value=5, help="1 = Very Poor, 10 = Very Excellent quality of materials and finish.")
        overall_cond = st.slider("Overall House Condition", min_value=1, max_value=9, value=5, help="1 = Very Poor condition, 9 = Excellent condition.")
        
    with col3:
        year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2000, help="The original year the house was built.")
        year_remod_add = st.number_input("Year Remodeled", min_value=1950, max_value=2026, value=2000, help="Year of last remodeling or addition (if none, same as build year).")

    st.divider()

    # Section 2: Area & Living Space Features
    st.subheader("2. Room & Floor Sizes")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bsmt_fin_sf1 = st.number_input("Finished Basement Area (sq ft)", min_value=0.0, value=500.0, step=50.0)
        bsmt_unf_sf = st.number_input("Unfinished Basement Area (sq ft)", min_value=0.0, value=200.0, step=50.0)
        total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0.0, value=700.0, step=50.0)
    with col2:
        first_flr_sf = st.number_input("1st Floor Area (sq ft)", min_value=0.0, value=850.0, step=50.0)
        second_flr_sf = st.number_input("2nd Floor Area (sq ft)", min_value=0.0, value=0.0, step=50.0)
        gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=0.0, value=1500.0, help="Total living area above ground level.")
    with col3:
        wood_deck_sf = st.number_input("Wooden Deck Area (sq ft)", min_value=0.0, value=0.0, step=10.0)
        open_porch_sf = st.number_input("Open Porch Area (sq ft)", min_value=0.0, value=0.0, step=10.0)
        enclosed_porch = st.number_input("Enclosed Porch Area (sq ft)", min_value=0.0, value=0.0, step=10.0)

    st.divider()

    # Section 3: Rooms, Bathrooms & Features
    st.subheader("3. Rooms, Bathrooms & Garage")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bsmt_full_bath = st.number_input("Basement Full Bathrooms", min_value=0, max_value=3, value=1)
        full_bath = st.number_input("Full Bathrooms (Above Ground)", min_value=0, max_value=4, value=2)
        half_bath = st.number_input("Half Bathrooms (Above Ground)", min_value=0, max_value=2, value=0)
    with col2:
        bedroom_abv_gr = st.number_input("Bedrooms (Above Ground)", min_value=0, max_value=8, value=3)
        kitchen_abv_gr = st.number_input("Kitchens (Above Ground)", min_value=0, max_value=3, value=1)
        tot_rms_abv_grd = st.number_input("Total Rooms (Above Ground)", min_value=2, max_value=14, value=6, help="Total rooms excluding bathrooms.")
    with col3:
        fireplaces = st.number_input("Number of Fireplaces", min_value=0, max_value=4, value=0)
        garage_cars = st.number_input("Garage Capacity (Car Spaces)", min_value=0, max_value=5, value=2, help="How many cars fit in the garage.")
        garage_area = st.number_input("Garage Area (sq ft)", min_value=0.0, value=500.0, step=50.0)

    st.divider()

    # Section 4: Miscellaneous & Sale Details
    st.subheader("4. Additional Features & Sale Info")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        three_ssn_porch = st.number_input("3-Season Porch (sq ft)", min_value=0.0, value=0.0, step=10.0)
        screen_porch = st.number_input("Screen Porch (sq ft)", min_value=0.0, value=0.0, step=10.0)
    with col2:
        pool_area = st.number_input("Swimming Pool Area (sq ft)", min_value=0.0, value=0.0, step=10.0)
        misc_val = st.number_input("Misc Feature Value ($)", min_value=0.0, value=0.0, step=100.0, help="Value of any other miscellaneous features.")
    with col3:
        mo_sold = st.slider("Month Sold", min_value=1, max_value=12, value=6, help="1 = January, 12 = December")
    with col4:
        yr_sold = st.number_input("Year Sold", min_value=2000, max_value=2030, value=2008)

    st.markdown("")
    submitted = st.form_submit_button("Predict House Price", type="primary", use_container_width=True)

# Process form submission
if submitted:
    # Compile inputs into a DataFrame matching all training feature names exactly
    input_df = pd.DataFrame({
        'MSSubClass': [mssubclass],
        'LotArea': [lot_area],
        'OverallQual': [overall_qual],
        'OverallCond': [overall_cond],
        'YearBuilt': [year_built],
        'YearRemodAdd': [year_remod_add],
        'BsmtFinSF1': [bsmt_fin_sf1],
        'BsmtUnfSF': [bsmt_unf_sf],
        'TotalBsmtSF': [total_bsmt_sf],
        '1stFlrSF': [first_flr_sf],
        '2ndFlrSF': [second_flr_sf],
        'GrLivArea': [gr_liv_area],
        'BsmtFullBath': [bsmt_full_bath],
        'FullBath': [full_bath],
        'HalfBath': [half_bath],
        'BedroomAbvGr': [bedroom_abv_gr],
        'KitchenAbvGr': [kitchen_abv_gr],
        'TotRmsAbvGrd': [tot_rms_abv_grd],
        'Fireplaces': [fireplaces],
        'GarageCars': [garage_cars],
        'GarageArea': [garage_area],
        'WoodDeckSF': [wood_deck_sf],
        'OpenPorchSF': [open_porch_sf],
        'EnclosedPorch': [enclosed_porch],
        '3SsnPorch': [three_ssn_porch],
        'ScreenPorch': [screen_porch],
        'PoolArea': [pool_area],
        'MiscVal': [misc_val],
        'MoSold': [mo_sold],
        'YrSold': [yr_sold]
    })
    
    
    
    # Make prediction using the loaded model
    prediction = model.predict(input_df)

    st.metric(label="Estimated House Sale Price", value=f"${prediction[0]:,.2f}")