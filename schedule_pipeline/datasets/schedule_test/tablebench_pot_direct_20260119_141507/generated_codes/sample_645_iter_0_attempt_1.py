import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'sales (billion )' and 'profits (billion )'
correlation = df['sales (billion )'].corr(df['profits (billion )'])
print(f"Final Answer: {correlation:.2f}")