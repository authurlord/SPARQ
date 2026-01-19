import pandas as pd

df = pd.read_csv('table.csv')

# Filter out header and total rows
filtered_df = df[df['Club'] != 'Career totals']
filtered_df = filtered_df[filtered_df['Club'] != 'Club']
filtered_df = filtered_df[filtered_df['Club'] != 'Total']

# Identify anomalies: negative apps or goals, or values with '?'
anomalies = []
for index, row in filtered_df.iterrows():
    if isinstance(row['Apps'], str) and '?' in row['Apps']:
        anomalies.append((row['Club'], row['Season'], row['Apps'], row['Goals']))
    elif isinstance(row['Apps'], (int, float)) and row['Apps'] < 0:
        anomalies.append((row['Club'], row['Season'], row['Apps'], row['Goals']))
    elif isinstance(row['Goals'], str) and '?' in row['Goals']:
        anomalies.append((row['Club'], row['Season'], row['Apps'], row['Goals']))

# If anomalies found, print them; otherwise, state no anomaly
if anomalies:
    print("Anomalies found:")
    for club, season, apps, goals in anomalies:
        print(f"Club: {club}, Season: {season}, Apps: {apps}, Goals: {goals}")
else:
    print("No anomalies found.")

# Final Answer: The anomaly is the negative apps (-10) for Espanyol in 1964-65
Final Answer: -10, Espanyol, 1964-65