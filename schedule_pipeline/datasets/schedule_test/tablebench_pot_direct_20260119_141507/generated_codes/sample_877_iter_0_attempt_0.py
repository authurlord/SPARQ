import pandas as pd

df = pd.read_csv('table.csv')
# Convert the temperature columns to numeric
df['january (avg high degree)'] = pd.to_numeric(df['january (avg high degree)'], errors='coerce')
df['july (avg high degree)'] = pd.to_numeric(df['july (avg high degree)'], errors='coerce')

# Calculate the mean average high temperatures for January and July
mean_jan_high = df['january (avg high degree)'].mean()
mean_jul_high = df['july (avg high degree)'].mean()

# Compute the difference
temp_difference = mean_jul_high - mean_jan_high
print(f"Final Answer: {temp_difference:.1f}")