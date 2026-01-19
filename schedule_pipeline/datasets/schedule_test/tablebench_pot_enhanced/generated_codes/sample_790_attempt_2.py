import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'average' column to numeric, handling 'n/a' values
df['average'] = pd.to_numeric(df['average'], errors='coerce')
# Calculate standard deviation of the average column
std_average = df['average'].std()
print(f"Final Answer: {std_average:.2f}")