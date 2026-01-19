import pandas as pd

df = pd.read_csv('table.csv')

# Identify rows with negative Apps or question marks
anomalies = df[(df['Apps'].astype(str).str.contains('-|\\?')) | (df['Goals'].astype(str).str.contains('\\?'))]

# Filter out the header and total rows
anomalies = anomalies[anomalies['Club'] != 'Career totals']
anomalies = anomalies[anomalies['Club'] != 'Club']
anomalies = anomalies[anomalies['Club'] != 'Real Madrid']

# Check for negative Apps in specific row
negative_app_row = df[(df['Club'] == 'Espanyol') & (df['Season'] == '1964-65') & (df['Apps'] == '-10')]

if not negative_app_row.empty:
    print(f"Final Answer: Espanyol 1964-65 with -10 Apps")
else:
    print(f"Final Answer: No clear anomaly found")