import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where cost <= 1400
filtered_df = df[df['cost'] <= '1400 / kwp']
# Define the minimum kwh/kwp threshold
min_kwh = 2000
# Get columns with kwh/kwp >= 2000
cols_to_sum = [col for col in df.columns if col.replace(' ', '').replace('kwh', '').replace('/kwp', '').replace('y', '').isdigit() and int(col.replace(' ', '').replace('kwh', '').replace('/kwp', '').replace('y', '')) >= min_kwh]
# Convert filtered_df to numeric for summing
filtered_df[cols_to_sum] = filtered_df[cols_to_sum].astype(float)
# Calculate total cost
total_cost = filtered_df[cols_to_sum].sum().sum()
print(f"Final Answer: {total_cost}")