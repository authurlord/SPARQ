import pandas as pd

df = pd.read_csv('table.csv')
# Convert '1948' column to integer for proper sorting and summing
df['1948'] = df['1948'].astype(int)
# Sort by '1948' in descending order and take top 5
top_5_population_1948 = df.nlargest(5, '1948')['1948'].sum()
print(f"Final Answer: {top_5_population_1948}")