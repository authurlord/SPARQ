import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert all values in the data to numeric (in case of string numbers)
df = df.apply(pd.to_numeric, errors='coerce')

# Identify anomalies: values above 1000 or below 1 in any cell
anomalies = []
for col in df.columns[1:]:
    for idx, val in df[col].items():
        if val > 1000 or val < 1:
            anomalies.append((col, idx, val))

# Print detected anomalies
print(f"Final Answer: {anomalies}")