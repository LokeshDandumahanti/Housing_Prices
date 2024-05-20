import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the model and columns
lr_clf = joblib.load("C:/Users/vijay/OneDrive/Desktop/Banglore Housing Project/banglore_home_prices_model.pkl")
X_columns = pd.read_csv("C:/Users/vijay/OneDrive/Desktop/Banglore Housing Project/dora.csv")
OHE = pd.read_csv("C:/Users/vijay/OneDrive/Desktop/Banglore Housing Project/B5.csv")
locations = OHE['location'].tolist()

# Non-changeable variables
bhk1 = 5
bath1 = 5

def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X_columns.columns == location)[0][0]

    x = np.zeros(len(X_columns.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

def get_price_predictions(location, sqft, bhk):
    all_predictions = []
    for bhk_val in range(1, bhk+1):
        predictions = []
        for bath in range(1, 6):
            price_prediction = predict_price(location, sqft, bath, bhk_val)
            predictions.append(price_prediction)
        all_predictions.append(predictions)
    return all_predictions

st.title('House Price Prediction')

# Sidebar with area and location selection
sqft = st.sidebar.slider('Select the area in sq meters:', min_value=500.0, max_value=3000.0, value=500.0)
location = st.sidebar.selectbox('Select a location:', locations)
bhk = st.sidebar.slider('Select BHK (1-5):', min_value=1, max_value=5)
bath = st.sidebar.slider('Select Bathrooms (1-5):', min_value=1, max_value=5)

estimated_price = predict_price(location, sqft, bath, bhk)
st.write(f"Estimated Price per sqft : ₹ {estimated_price}")

# Predict prices for different numbers of BHKs
predictions = get_price_predictions(location, sqft, bhk1)

# Display a spreadsheet-like table of prices
prices_table = pd.DataFrame(predictions, columns=[f"{i+1} BHK" for i in range(bhk1)], index=[f"{i} Bathrooms" for i in range(1, bath1+1)])
st.table(prices_table)

# Plot graphs for each number of BHKs
fig, axs = plt.subplots(bhk1, 1, figsize=(10, bhk1*5), sharex=True)
bath_values = range(1, 6)
colors = ['blue', 'green', 'red', 'purple', 'orange']  # Define different colors for each BHK

for i in range(bhk1):
    axs[i].plot(bath_values, predictions[i], label=f'{i+1} BHK', color=colors[i])  # Use a different color for each BHK
    axs[i].set_ylabel('Predicted Price per sqft (in ₹)')
    axs[i].set_title(f'Predicted Price for {i+1} BHK (in ₹)')
    axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Position legend to the right of the graph

# Set common x-axis label
fig.text(0.5, 0.04, 'Number of Bathrooms', ha='center', va='center')

plt.tight_layout(pad=3.0)
st.pyplot(fig)
