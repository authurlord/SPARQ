import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Percent Increase (Decrease)' to numeric, handling non-numeric values like 'nm'
df['Percent Increase (Decrease)'] = pd.to_numeric(df['Percent Increase (Decrease)'], errors='coerce')

# Filter rows where the percentage increase is greater than 5
high_increase = df[df['Percent Increase (Decrease)'] > 5]

# Count the number of such categories
count = len(high_increase)

print(f"Final Answer: {count}")