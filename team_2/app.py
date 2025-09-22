import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
with open("best_model.pkl", "rb") as f:
    car_model = pickle.load(f)

# Load dataset
data = pd.read_csv("cleaned_cars_data.csv")

st.set_page_config(page_title="Car Price Prediction", page_icon="🚘", layout="wide")

st.title("🚘 Car Price Prediction App")
st.write("Explore car datasets and predict prices using the trained car_model.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Predict", "📈 Insights"])

# Dashboard
with tab1:
    st.subheader("Dataset Overview")
    st.write(data.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cars", len(data))
    col2.metric("Avg Price (OMR)", f"{data['Price'].mean():,.0f}")
    col3.metric("Top Brand", data['Brand'].mode()[0])

    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data["Price"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

# Prediction
with tab2:
    st.subheader("Enter Car Details")
    brand = st.selectbox("Brand", sorted(data["Brand"].dropna().unique()))
    year = st.slider("Year", 1990, 2025, 2018)
    kilometers = st.number_input("Kilometers", 0, 500000, 50000)

    if st.button("Predict Price 🚀"):
        input_df = pd.DataFrame({
            "Brand": [brand],
            "Year": [year],
            "Kilometers": [kilometers]
        })

        with st.spinner("Calculating..."):
            price = car_model.predict(input_df[["Year", "Kilometers"]])[0]

        st.success(f"💰 Estimated Price: **{price:,.2f} OMR**")

# Insights
with tab3:
    st.subheader("Correlation Heatmap")
    corr = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
