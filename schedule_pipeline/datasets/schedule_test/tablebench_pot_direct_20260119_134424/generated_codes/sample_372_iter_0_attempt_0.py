import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Percent Increase (Decrease)' to numeric, handling parentheses for negative values
df['Percent Increase (Decrease)'] = pd.to_numeric(df['Percent Increase (Decrease)'].str.replace('nm (1)', '0').str.replace('—', '0').str.strip('()'), errors='coerce')

# Filter rows where percentage increase > 5
count_high_increase = (df['Percent Increase (Decrease)'] > 5).sum()

print(f"Final Answer: {count_high_increase}")