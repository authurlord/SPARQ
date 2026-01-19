import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'property taxes' to numeric
df['property taxes'] = pd.to_numeric(df['property taxes'])
# Calculate the average annual increase
start_value = df[df['year'] == '2000']['property taxes'].values[0]
end_value = df[df['year'] == '2005']['property taxes'].values[0]
average_increase = (end_value - start_value) / 5
print(f"Final Answer: {average_increase:.0f}")