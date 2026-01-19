import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'property taxes' to numeric
df['property taxes'] = pd.to_numeric(df['property taxes'])
# Calculate the total increase over the period
total_increase = df.loc[df['year'] == '2005', 'property taxes'].values[0] - df.loc[df['year'] == '2000', 'property taxes'].values[0]
# Calculate average annual increase
avg_annual_increase = total_increase / 5
print(f"Final Answer: {avg_annual_increase:.0f}")