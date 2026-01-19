import pandas as pd

df = pd.read_csv('table.csv')
# Filter earthquakes with magnitude >= 6.5 and count them
count_high_magnitude = df[df['Magnitude'] >= '6.5'].shape[0]
print(f"Final Answer: {count_high_magnitude}")