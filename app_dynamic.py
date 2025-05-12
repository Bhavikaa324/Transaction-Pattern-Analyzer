import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import time
import requests

# Function to load data dynamically (this could be from an API or database)
@st.cache_data
def load_data():
    # Example: Replace this with the actual method of fetching live data (API, streaming, or database)
    df = pd.read_csv('PS_20174392719_1491204439457_log.csv')  # Update this path as needed
    df = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']]
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    df.dropna(inplace=True)
    return df

# Function to simulate live data fetching
def fetch_live_data():
    # Simulate the addition of new data (replace with actual live data source)
    new_data = pd.read_csv('D:/transaction analyzer/PS_20174392719_1491204439457_log.csv')  # Replace with actual API call or DB fetch
    return new_data

# Streamlit app title
st.title("💳 Dynamic Transaction Pattern Analyzer")

# Sidebar - options to set the number of clusters dynamically
st.sidebar.header("Clustering Settings")
num_clusters = st.sidebar.slider("Select number of clusters", 2, 10, 4)

# Load and preprocess data dynamically (refresh every time the user presses the button or on new data)
if st.button('Refresh Data'):
    df = fetch_live_data()
    st.write("Data Refreshed!")
else:
    df = load_data()  # Use cached data for efficiency

# Standardizing the data for clustering
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

# KMeans clustering with dynamic number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
df['Cluster'] = kmeans.fit_predict(scaled)

# Display cluster summary
st.subheader("📊 Cluster Summary")
cluster_summary = df.groupby('Cluster')[['amount', 'oldbalanceOrg', 'newbalanceOrig']].mean().round(2)
st.dataframe(cluster_summary)

# Bar chart: Average Transaction Amount per Cluster
st.subheader("💰 Average Transaction Amount per Cluster")
fig1 = px.bar(cluster_summary, y='amount', title='Avg Transaction Amount per Cluster')
st.plotly_chart(fig1)

# Scatter plot: Visualize clusters
st.subheader("🔍 Cluster Scatterplot")
fig2 = px.scatter(df, x='oldbalanceOrg', y='newbalanceOrig', color='Cluster',
                  title='Cluster Visualization by Balance')
st.plotly_chart(fig2)

# Optional: Download button for clustered data
st.subheader("⬇️ Download Clustered Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "clustered_transactions.csv", "text/csv")

# Refresh the page every 30 seconds for new data (optional, to simulate a dynamic update in live data)
st.write("Waiting for new data updates...")
time.sleep(30)  # This will refresh the app every 30 seconds (for simulation)

# Automatically fetch new data (you can integrate with a real-time data stream/API)
df = fetch_live_data()

