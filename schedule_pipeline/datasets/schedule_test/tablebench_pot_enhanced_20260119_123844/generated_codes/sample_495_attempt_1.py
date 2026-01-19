import pandas as pd

df = pd.read_csv('table.csv')
# Clean 'capacity in use' column: remove spaces and commas, convert to float
df['capacity in use'] = df['capacity in use'].str.replace(' , ', '.', regex=False).str.replace('%', '', regex=False).astype(float)
# Find the row with the maximum capacity in use
max_utilization_row = df.loc[df['capacity in use'].idxmax()]
# Extract the location
highest_utilization_airport = max_utilization_row['location']
print(f"Final Answer: {highest_utilization_airport}")