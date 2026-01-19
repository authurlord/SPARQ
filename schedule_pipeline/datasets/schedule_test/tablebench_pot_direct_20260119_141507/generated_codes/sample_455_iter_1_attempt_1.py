import pandas as pd

# Load the dataset
df = pd.read_csv('table.csv')

# Identify anomalies
anomalies = []

# Check for invalid 'whenbuilt' values
invalid_whenbuilt = df[df['whenbuilt'].str.contains(r'\D', na=False) | (df['whenbuilt'].astype(str).str.match(r'^\d{4}$')) == False]
for idx, row in invalid_whenbuilt.iterrows():
    anomalies.append({
        'name': row['name'],
        'whenbuilt': row['whenbuilt'],
        'withdrawn': row['withdrawn']
    })

# Check for withdrawal before built year
df['whenbuilt_numeric'] = pd.to_numeric(df['whenbuilt'], errors='coerce')
df['withdrawn_numeric'] = pd.to_numeric(df['withdrawn'], errors='coerce')
invalid_withdrawal = df[(df['withdrawn_numeric'] < df['whenbuilt_numeric']) & (pd.notna(df['whenbuilt_numeric']) & pd.notna(df['withdrawn_numeric']))]
for idx, row in invalid_withdrawal.iterrows():
    anomalies.append({
        'name': row['name'],
        'whenbuilt': row['whenbuilt'],
        'withdrawn': row['withdrawn']
    })

# Remove duplicates if any
anomalies = list(dict.fromkeys(anomalies))  # Keep first occurrence

# Print anomalies
if anomalies:
    for anomaly in anomalies:
        print(f"Anomaly: {anomaly['name']} - Built: {anomaly['whenbuilt']}, Withdrawn: {anomaly['withdrawn']}")
else:
    print("No anomalies found.")

Final Answer: 601 squadron, 257 squadron, 249 squadron, 46 squadron, 264 squadron, 41 squadron, 603 squadron, 222 squadron, 141 squadron, 92 squadron, 615 squadron, 605 squadron, 253 squadron, 501 squadron, 145 squadron, 219 squadron, 213 squadron, 602 squadron, weymouth, city of wells, saunton, mortehoe, brentor, trevone, holsworthy, templecombe, lynmouth, appledore, hartland, lapford, calstock, bere alston, swanage, lydford, blandford forum, wincanton, sir trafford leigh - mallory, 66 squadron