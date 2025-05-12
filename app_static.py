import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
    df = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']]
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    df.dropna(inplace=True)
    return df

df = load_data()
st.title("💳 Transaction Pattern Analyzer")

# Sidebar - cluster options
st.sidebar.header("Clustering Settings")
num_clusters = st.sidebar.slider("Select number of clusters", 2, 10, 4)

# Standardize
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

# KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
df['Cluster'] = kmeans.fit_predict(scaled)

# Cluster summary
st.subheader("📊 Cluster Summary")
cluster_summary = df.groupby('Cluster')[['amount', 'oldbalanceOrg', 'newbalanceOrig']].mean().round(2)
st.dataframe(cluster_summary)

# Bar chart: Avg Transaction Amount
st.subheader("💰 Average Transaction Amount per Cluster")
fig1 = px.bar(cluster_summary, y='amount', title='Avg Transaction Amount per Cluster')
st.plotly_chart(fig1)

# Scatterplot: Clusters
st.subheader("🔍 Cluster Scatterplot")
fig2 = px.scatter(df, x='oldbalanceOrg', y='newbalanceOrig', color='Cluster',
                  title='Cluster Visualization by Balance')
st.plotly_chart(fig2)

# Optional: Download clustered dataset
st.subheader("⬇️ Download Clustered Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "clustered_transactions.csv", "text/csv")
