import pandas as pd

df = pd.read_csv('table.csv')
# Find the row with the minimum local magnitude
min_magnitude_row = df[df['local magnitude'] == df['local magnitude'].min()]
# Extract the origin time
lowest_magnitude_time = min_magnitude_row['origin time'].values[0]
print(f"Final Answer: {lowest_magnitude_time}")