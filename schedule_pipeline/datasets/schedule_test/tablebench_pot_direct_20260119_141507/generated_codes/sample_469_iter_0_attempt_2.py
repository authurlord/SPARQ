import pandas as pd

df = pd.read_csv('table.csv')

# Convert all columns to numeric where possible, handling non-numeric values
df_numeric = df.copy()
for col in df_numeric.columns:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

# Check for discrepancies in total deaths (military + civilian vs total deaths)
anomalies = []

# Check if total deaths (not including foreigners) matches sum of military and civilian deaths
for idx, row in df_numeric.iterrows():
    military = row['military deaths']
    civilian = row['civilian deaths']
    total_deaths = row['total deaths (not including foreigners)']
    
    # Skip if any value is NaN
    if pd.isna(military) or pd.isna(civilian) or pd.isna(total_deaths):
        continue
        
    # Check if sum of military and civilian deaths is close to total deaths
    if abs(military + civilian - total_deaths) > 100:  # Threshold for anomaly
        anomalies.append(f"Row {idx}: Military={military}, Civilian={civilian}, Total={total_deaths} — sum mismatch")
    
    # Check for extreme values
    if military > 100000 or civilian > 100000 or total_deaths > 100000:
        anomalies.append(f"Row {idx}: High values — Military={military}, Civilian={civilian}, Total={total_deaths}")

# Also check for "unknown" or inconsistent entries
for idx, row in df.iterrows():
    if row['military deaths'] == 'unknown' or row['civilian deaths'] == 'unknown' or row['total deaths (not including foreigners)'] == 'unknown':
        anomalies.append(f"Row {idx}: Contains 'unknown' in key fields")

# Remove duplicates and print
unique_anomalies = list(set(anomalies))
print(f"Final Answer: {', '.join(unique_anomalies)}")