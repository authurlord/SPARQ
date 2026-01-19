import pandas as pd

df = pd.read_csv('table.csv')
# Clean 'capacity in use' column: remove comma and '%' sign, convert to float
df['capacity in use'] = df['capacity in use'].str.replace(',', '').str.replace('%', '').astype(float)
# Find the index of the maximum utilization rate
max_index = df['capacity in use'].idxmax()
# Get the location (airport) with the highest utilization rate
highest_utilization_airport = df.loc[max_index, 'location']
print(f"Final Answer: {highest_utilization_airport}")