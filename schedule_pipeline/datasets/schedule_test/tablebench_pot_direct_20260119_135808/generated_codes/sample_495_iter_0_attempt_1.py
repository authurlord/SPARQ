import pandas as pd

df = pd.read_csv('table.csv')
# Clean the 'capacity in use' column by removing spaces, commas, and '%', then convert to float
df['capacity in use'] = df['capacity in use'].str.replace(' , ', '.', regex=False).str.replace('%', '', regex=False).astype(float)
# Find the location with the highest capacity utilization rate
max_utilization_airport = df.loc[df['capacity in use'].idxmax(), 'location']
print(f"Final Answer: {max_utilization_airport}")