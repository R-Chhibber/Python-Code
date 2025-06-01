import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set Streamlit layout
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🛍️ Customer Segmentation using K-Means")

# Sidebar Parameters
st.sidebar.header("Generate Customer Data")
n_customers = st.sidebar.slider("Number of Customers", 50, 1000, 200)

# Generate synthetic customer dataset
np.random.seed(42)
data = {
    'CustomerID': np.arange(1, n_customers + 1),
    'AnnualIncome': np.random.normal(60000, 15000, n_customers).astype(int),
    'SpendingScore': np.random.normal(50, 25, n_customers).clip(1, 100).astype(int)
}
df = pd.DataFrame(data)

# Show data sample
if st.checkbox("Show Raw Data"):
    st.write(df.head())

# Preprocessing
features = df[['AnnualIncome', 'SpendingScore']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Elbow method to find optimal k
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
st.subheader("📈 Elbow Method to Determine Optimal Number of Clusters")
fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss, marker='o')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('WCSS (Inertia)')
ax.set_title('Elbow Method')
st.pyplot(fig)

# Choose number of clusters based on elbow
k = st.slider("Select Number of Clusters (k)", 2, 10, 4)
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Plot Clusters
st.subheader("🧠 Customer Clusters")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x='AnnualIncome', y='SpendingScore', hue='Cluster', palette='tab10', ax=ax2)
ax2.set_title("Customer Segments")
ax2.set_xlabel("Annual Income ($)")
ax2.set_ylabel("Spending Score")
st.pyplot(fig2)

# Show Cluster Stats
st.subheader("📊 Cluster Summary")
st.dataframe(df.groupby('Cluster')[['AnnualIncome', 'SpendingScore']].mean().round(2))
