import pandas as pd

df = pd.read_csv('table.csv')
# Extract average viewers column
avg_viewers = df['average viewers (millions)'].astype(float)

# Calculate Q1, Q3, and IQR
Q1 = avg_viewers.quantile(0.25)
Q3 = avg_viewers.quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = avg_viewers[(avg_viewers < lower_bound) | (avg_viewers > upper_bound)]

# If any outliers exist, return the corresponding series name
if not outliers.empty:
    anomaly_series = df[df['average viewers (millions)'].astype(float).isin(outliers)].iloc[0]['series']
    print(f"Final Answer: {anomaly_series}")
else:
    print("Final Answer: no anomaly")