import pandas as pd

df = pd.read_csv('table.csv')
# The column 'weight (kg / m)' contains numerical values
mean_weight = df['weight (kg / m)'].mean()
print(f"Final Answer: {mean_weight:.1f}")