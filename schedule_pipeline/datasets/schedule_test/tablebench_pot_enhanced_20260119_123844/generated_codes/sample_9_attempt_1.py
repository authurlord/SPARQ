import pandas as pd

df = pd.read_csv('table.csv')
# The column 'podiums' contains numerical values
mean_podiums = df['podiums'].mean()
print(f"Final Answer: {mean_podiums:.1f}")