import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Magnitude' column to float for numerical comparison
df['Magnitude'] = pd.to_numeric(df['Magnitude'])
# Count earthquakes with magnitude >= 6.5
count_high_magnitude = df[df['Magnitude'] >= 6.5].shape[0]
print(f"Final Answer: {count_high_magnitude}")