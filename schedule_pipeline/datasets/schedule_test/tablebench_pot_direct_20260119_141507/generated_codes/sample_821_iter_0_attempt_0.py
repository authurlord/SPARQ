import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation coefficient between 'draw' and 'points'
correlation = df['draw'].corr(df['points'])
print(f"Final Answer: {correlation:.2f}")