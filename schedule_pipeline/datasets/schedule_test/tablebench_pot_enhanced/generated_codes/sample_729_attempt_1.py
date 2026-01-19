import pandas as pd

df = pd.read_csv('table.csv')
# Find the row with the minimum local magnitude
min_magnitude_row = df.loc[df['local magnitude'].idxmin()]
# Extract the origin time
lowest_magnitude_time = min_magnitude_row['origin time']
print(f"Final Answer: {lowest_magnitude_time}")