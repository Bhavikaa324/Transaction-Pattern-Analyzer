# Transaction-Pattern-Analyzer
This Streamlit app analyzes transaction data by clustering transactions based on different features such as amount, oldbalanceOrg, and newbalanceOrig. The app allows dynamic clustering, where the user can adjust the number of clusters and refresh the data to analyze the latest patterns.

Features
1.Dynamic Clustering: Choose the number of clusters (from 2 to 10) via the sidebar.

2.Live Data Fetching: Click the "Refresh Data" button to fetch the latest data.

3.Cluster Analysis: Visualize the average transaction amount per cluster and explore how clusters relate to balances.

4.Scatter Plot: Visualize clusters based on oldbalanceOrg vs. newbalanceOrig.

5.Download: Download the clustered data as a CSV file.

6.Data Refresh: Automatically refresh the page every 30 seconds for live data updates.