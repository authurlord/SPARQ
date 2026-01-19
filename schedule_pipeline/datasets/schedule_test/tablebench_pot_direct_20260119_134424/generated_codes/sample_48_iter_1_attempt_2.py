import pandas as pd

# Load the data from the JSON table
df = pd.read_csv('table.csv')

# Convert 'points' column to numeric, handling any non-numeric values
df['points'] = pd.to_numeric(df['points'], errors='coerce')

# Calculate the average number of points
average_points = df['points'].mean()

# Print the final answer
print(f"Final Answer: {average_points:.1f}")