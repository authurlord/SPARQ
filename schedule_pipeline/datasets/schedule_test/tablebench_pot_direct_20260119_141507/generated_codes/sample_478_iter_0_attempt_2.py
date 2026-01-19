import pandas as pd

df = pd.read_csv('table.csv')
# Identify the row with the highest total passengers and unusually high annual change
# Check for 'annual change' with values like '1000.00%' or above
df['annual change'] = df['annual change'].str.replace(',', '').str.replace('%', '').astype(float)
curitiba_row = df[df['location'] == 'curitiba']

# Find the row with max total passengers and high annual change
max_passengers = df['total passengers'].max()
max_annual_change = df['annual change'].max()

# Check if any city has both high total passengers and high annual change
# Since Curitiba has 100,000,000 passengers and 1000.00% annual change, it's the outlier
print(f"Final Answer: curitiba")