import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where cost per kwp is at most $1400
filtered_df = df[df['cost'] <= '1400 / kwp']

# Define the production rate columns of interest (>= 2000 kwh/kwp/y)
production_columns = ['2000 kwh / kwp y', '1800 kwh / kwp y', '1600 kwh / kwp y', '1400 kwh / kwp y', '1200 kwh / kwp y', '1000 kwh / kwp y', '800 kwh / kwp y']
# But we want only those >= 2000 kwh/kwp/y, so only '2000 kwh / kwp y' and above
# Actually, the headers are in descending order. So '2000 kwh / kwp y' and above means only '2000 kwh / kwp y' and higher
# But looking at data: '2400', '2200', '2000', '1800', ... — so we need columns: '2400', '2200', '2000'

# Correct: We want columns with values >= 2000 kwh/kwp/y → columns: '2400 kwh / kwp y', '2200 kwh / kwp y', '2000 kwh / kwp y'
target_columns = ['2400 kwh / kwp y', '2200 kwh / kwp y', '2000 kwh / kwp y']

# Extract the values from these columns for the filtered rows
values = filtered_df[target_columns].stack().astype(float)

# Sum all the values
total_cost = values.sum()

print(f"Final Answer: {total_cost}")