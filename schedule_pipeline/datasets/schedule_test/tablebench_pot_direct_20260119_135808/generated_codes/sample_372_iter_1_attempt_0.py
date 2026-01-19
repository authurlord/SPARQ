import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Percent Increase (Decrease)' to numeric, coercing errors to NaN
df['Percent Increase (Decrease) (In cents, except percentage changes)'] = pd.to_numeric(df['Percent Increase (Decrease) (In cents, except percentage changes)'], errors='coerce')
# Drop rows where the percentage change is not a valid number (like 'nm (1)')
df_filtered = df.dropna(subset=['Percent Increase (Decrease) (In cents, except percentage changes)'])
# Count how many have percentage increase > 5
count = df_filtered[df_filtered['Percent Increase (Decrease) (In cents, except percentage changes)'] > 5].shape[0]
print(f"Final Answer: {count}")