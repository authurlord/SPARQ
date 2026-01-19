import pandas as pd

df = pd.read_csv('table.csv')
# Clean the 'capacity in use' column: remove commas and %, convert to float
df['capacity in use'] = df['capacity in use'].str.replace(',', '').str.replace('%', '').astype(float)
# Find the index of the maximum capacity in use
max_index = df['capacity in use'].idxmax()
# Get the location of the airport with the highest utilization rate
highest_utilization_airport = df.loc[max_index, 'location']
print(f"Final Answer: {highest_utilization_airport}")