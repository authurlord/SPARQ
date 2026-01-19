import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Magnitude' column to float for numerical comparison
df['Magnitude'] = pd.to_numeric(df['Magnitude'])
# Filter rows with magnitude >= 6.5
high_magnitude_earthquakes = df[df['Magnitude'] >= 6.5]
# Count the number of such earthquakes
count = len(high_magnitude_earthquakes)
print(f"Final Answer: {count}")