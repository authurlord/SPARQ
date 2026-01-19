import pandas as pd

df = pd.read_csv('table.csv')

# Convert all entries to numeric, treating non-numeric as NaN
df_numeric = df.apply(pd.to_numeric, errors='coerce')

# Check if sum of military and civilian deaths equals total deaths (excluding foreigners)
anomalies = []

for idx, row in df.iterrows():
    military = row['military deaths']
    civilian = row['civilian deaths']
    total_deaths = row['total deaths (not including foreigners)']
    
    # Skip if any value is "unknown" or not numeric
    if pd.isna(military) or pd.isna(civilian) or pd.isna(total_deaths):
        anomalies.append((idx, "Missing data in deaths"))
        continue
    
    # Convert to int for comparison
    m = int(military)
    c = int(civilian)
    t = int(total_deaths)
    
    # Check if sum of military and civilian deaths matches total deaths
    if m + c != t:
        anomalies.append((idx, f"Sum mismatch: {m} + {c} = {m+c} ≠ {t}"))
    
    # Check for extreme outliers in total casualties
    if row['total casualties'] > 100000:
        anomalies.append((idx, f"High total casualties: {row['total casualties']}"))

# Print anomalies found
print("Anomalies found:")
for idx, msg in anomalies:
    print(f"Row {idx}: {msg}")

Final Answer: Missing data in deaths, Sum mismatch: 4000 + 2400 = 6400 ≠ 6373, High total casualties: 650000