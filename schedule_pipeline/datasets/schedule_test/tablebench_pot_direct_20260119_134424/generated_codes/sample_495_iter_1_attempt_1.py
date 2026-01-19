import pandas as pd

df = pd.read_csv('table.csv')
# Clean 'capacity in use' by removing commas and converting to float
df['capacity in use'] = df['capacity in use'].str.replace(',', '').astype(float)
# Convert 'capacity' to integer
df['capacity'] = df['capacity'].astype(int)
# Calculate utilization rate
df['utilization_rate'] = (df['capacity in use'] / df['capacity']) * 100
# Find the airport with the highest utilization rate
max_utilization_airport = df.loc[df['utilization_rate'].idxmax()]['location']
print(f"Final Answer: {max_utilization_airport}")