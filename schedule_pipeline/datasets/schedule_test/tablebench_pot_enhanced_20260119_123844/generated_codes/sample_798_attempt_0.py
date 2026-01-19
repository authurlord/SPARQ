import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Conservative councillors' to integer
df['Conservative councillors'] = pd.to_numeric(df['Conservative councillors'])
# Extract values for 1947 and 1972
first_value = df[df['Year'] == '1947']['Conservative councillors'].values[0]
last_value = df[df['Year'] == '1972']['Conservative councillors'].values[0]
# Calculate average annual change
annual_change = (last_value - first_value) / (1972 - 1947)
print(f"Final Answer: {annual_change:.2f}")