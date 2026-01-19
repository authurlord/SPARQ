import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where cost <= 1400
filtered_df = df[df['cost'] <= '1400 / kwp']

# Select columns for production >= 2000 kWh/kwp/year
production_cols = ['2000 kwh / kwp y', '1800 kwh / kwp y', '1600 kwh / kwp y', '1400 kwh / kwp y', '1200 kwh / kwp y', '1000 kwh / kwp y', '800 kwh / kwp y']
# Note: We need to include only columns with production >= 2000, so we take from '2000' onward
valid_cols = ['2000 kwh / kwp y', '1800 kwh / kwp y', '1600 kwh / kwp y', '1400 kwh / kwp y', '1200 kwh / kwp y', '1000 kwh / kwp y', '800 kwh / kwp y']

# Convert the selected columns to numeric
filtered_df[valid_cols] = filtered_df[valid_cols].apply(pd.to_numeric)

# Sum all values in the filtered DataFrame
total_cost = filtered_df[valid_cols].sum().sum()

print(f"Final Answer: {total_cost}")