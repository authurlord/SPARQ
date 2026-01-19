import pandas as pd

df = pd.read_csv('table.csv')
# Extract urban percentage and compute average
urban_avg = df['urban , %'].mean()
print(f"Final Answer: {urban_avg:.1f}")