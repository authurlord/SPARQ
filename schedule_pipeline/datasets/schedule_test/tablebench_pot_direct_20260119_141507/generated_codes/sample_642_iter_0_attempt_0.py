import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'starting weight (kg)' and 'percentage lost'
correlation = df['starting weight (kg)'].corr(df['percentage lost'])
print(f"Final Answer: {correlation:.3f}")