import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = pd.to_numeric(df['issue price'])
# Calculate yearly increase
yearly_increase = df['issue price'].diff().dropna()
# Calculate average increase
avg_increase = yearly_increase.mean()
print(f"Final Answer: {avg_increase:.2f}")