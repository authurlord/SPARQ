import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'weight (kg / m)' column to numeric, handling any potential formatting issues
df['weight (kg / m)'] = pd.to_numeric(df['weight (kg / m)'], errors='coerce')
# Calculate the average weight, excluding any invalid entries
average_weight = df['weight (kg / m)'].mean()
print(f"Final Answer: {average_weight:.1f}")