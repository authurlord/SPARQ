import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the average weight
average_weight = df['weight (kg / m)'].mean()
print(f"Final Answer: {average_weight:.1f}")