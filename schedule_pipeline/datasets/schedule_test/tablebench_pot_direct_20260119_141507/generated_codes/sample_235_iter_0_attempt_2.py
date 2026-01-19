import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between upper index kcal/nm³ and upper index MJ/nm³
correlation = df['upper index kcal / nm 3'].corr(df['upper index mj / nm 3'])
print(f"Final Answer: {correlation:.3f}")