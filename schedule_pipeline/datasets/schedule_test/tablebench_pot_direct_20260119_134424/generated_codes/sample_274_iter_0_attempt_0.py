import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where cost is <= 1400
filtered_df = df[df['cost'] <= '1400 / kwp']
# Select columns for production >= 2000 kwh/kwp/y
columns_to_select = ['2000 kwh / kwp y', '1800 kwh / kwp y', '1600 kwh / kwp y', '1400 kwh / kwp y', '1200 kwh / kwp y', '1000 kwh / kwp y', '800 kwh / kwp y']
# Ensure correct column names are used
valid_columns = [col for col in columns_to_select if col in df.columns]
# Convert selected values to float and sum
total_cost = filtered_df[valid_columns].astype(float).sum().sum()
print(f"Final Answer: {total_cost}")