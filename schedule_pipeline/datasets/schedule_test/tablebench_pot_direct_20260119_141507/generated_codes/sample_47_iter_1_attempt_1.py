import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'weight (kg / m)' column and convert to numeric
weights = pd.to_numeric(df['weight (kg / m)'], errors='coerce')
# Calculate the mean of valid weights
average_weight = weights.mean()
print(f"Final Answer: {average_weight:.1f}")