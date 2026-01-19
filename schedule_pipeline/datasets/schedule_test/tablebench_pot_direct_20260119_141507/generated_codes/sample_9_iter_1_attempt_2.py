import pandas as pd

df = pd.read_csv('table.csv')
# Ensure 'podiums' is numeric and compute the mean
podiums = df['podiums'].astype(float).mean()
print(f"Final Answer: {podiums:.1f}")