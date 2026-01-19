import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points for' column to numeric, skipping the header row
points_for = pd.to_numeric(df['points for'], errors='coerce')
# Calculate standard deviation
std_points_for = points_for.std()
print(f"Final Answer: {std_points_for:.2f}")