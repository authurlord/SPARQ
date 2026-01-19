import pandas as pd

df = pd.read_csv('table.csv')
# Clean and convert 'capacity in use' column to float
df['capacity in use'] = df['capacity in use'].str.replace(',', '').str.replace('%', '').astype(float)
# Find the location with the highest capacity in use
max_utilization_airport = df.loc[df['capacity in use'].idxmax(), 'location']
print(f"Final Answer: {max_utilization_airport}")