import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points for' column to numeric, handling any non-numeric values if present
df['points for'] = pd.to_numeric(df['points for'], errors='coerce')
# Calculate standard deviation of 'points for'
std_points_for = df['points for'].std()
print(f"Final Answer: {std_points_for:.2f}")