import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Identify anomalies: negative values in 'Apps' column (impossible in football)
anomalies = df[(df['Apps'].astype(str).str.contains('-')) & (df['Club'] != 'Career totals')]

# Check if any negative values exist
if not anomalies.empty:
    print(f"Anomaly found: {anomalies[['Club', 'Season', 'Apps', 'Goals']]}")
else:
    print("No anomalies found in the data.")

# Final Answer: Espanyol 1964-65 -10 Apps
Final Answer: Espanyol 1964-65 -10 Apps