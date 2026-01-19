import pandas as pd

# Load the data from the JSON table
df = pd.read_csv('table.csv')

# Convert the 'weight (kg / m)' column to numeric, ensuring proper parsing
df['weight (kg / m)'] = pd.to_numeric(df['weight (kg / m)'], errors='coerce')

# Calculate the average weight
average_weight = df['weight (kg / m)'].mean()

# Output the final answer
print(f"Final Answer: {average_weight:.1f}")