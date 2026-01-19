import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where cost <= 1400
filtered_df = df[df['cost'] <= '1400 / kwp']
# Select columns for 2000 kwh/kwp/year and above
columns_to_sum = ['2000 kwh / kwp y', '1800 kwh / kwp y', '1600 kwh / kwp y', '1400 kwh / kwp y', '1200 kwh / kwp y', '1000 kwh / kwp y', '800 kwh / kwp y']
# Convert the selected columns to numeric
filtered_df[columns_to_sum] = filtered_df[columns_to_sum].apply(pd.to_numeric)
# Sum all values in the filtered data
total_cost = filtered_df[columns_to_sum].sum().sum()
print(f"Final Answer: {total_cost}")