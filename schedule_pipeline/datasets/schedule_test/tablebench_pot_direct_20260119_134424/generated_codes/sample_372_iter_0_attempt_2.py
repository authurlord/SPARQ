import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Percent Increase (Decrease)' to numeric, handling non-numeric values
df['Percent Increase (Decrease)'] = pd.to_numeric(df['Percent Increase (Decrease)'], errors='coerce')
# Drop rows where the value is 'nm (1)' or NaN
df_filtered = df[df['Percent Increase (Decrease)'] > 5]
count = len(df_filtered)
print(f"Final Answer: {count}")