import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the difference between property taxes and investment earnings
df['difference'] = df['property taxes'] - df['investment earnings']
# Find the year with the largest difference
max_difference_year = df.loc[df['difference'].idxmax(), 'year']
print(f"Final Answer: {max_difference_year}")