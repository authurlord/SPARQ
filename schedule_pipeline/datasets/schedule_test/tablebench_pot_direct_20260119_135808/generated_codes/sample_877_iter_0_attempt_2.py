import pandas as pd

df = pd.read_csv('table.csv')
# Convert temperature columns to numeric
df['january (avg high degree)'] = pd.to_numeric(df['january (avg high degree)'])
df['july (avg high degree)'] = pd.to_numeric(df['july (avg high degree)'])

# Calculate average high temperatures
mean_jan_high = df['january (avg high degree)'].mean()
mean_jul_high = df['july (avg high degree)'].mean()

# Calculate the difference
difference = mean_jul_high - mean_jan_high
print(f"Final Answer: {difference:.1f}")