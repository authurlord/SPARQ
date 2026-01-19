import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Magnitude' column to float for comparison
df['Magnitude'] = pd.to_numeric(df['Magnitude'])
# Filter earthquakes with magnitude 6.5 or higher
high_magnitude_earthquakes = df[df['Magnitude'] >= 6.5]
# Count the number of such earthquakes
count = len(high_magnitude_earthquakes)
print(f"Final Answer: {count}")