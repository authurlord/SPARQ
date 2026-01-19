import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'average' column to numeric, handling 'n/a' as NaN
df['average'] = pd.to_numeric(df['average'], errors='coerce')
# Calculate standard deviation of the average column
std_avg = df['average'].std()
print(f"Final Answer: {std_avg:.2f}")