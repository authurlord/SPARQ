import pandas as pd

df = pd.read_csv('table.csv')
# Convert '1948' column to numeric for proper sorting and summation
df['1948'] = pd.to_numeric(df['1948'])
# Sort by '1948' in descending order and take top 5
top_5_1948 = df.nlargest(5, '1948')['1948'].sum()
print(f"Final Answer: {top_5_1948}")