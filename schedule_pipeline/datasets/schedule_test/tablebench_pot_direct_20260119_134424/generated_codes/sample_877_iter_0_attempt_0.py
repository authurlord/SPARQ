import pandas as pd

df = pd.read_csv('table.csv')
# Convert the temperature columns to numeric
df['january (avg high degree)'] = pd.to_numeric(df['january (avg high degree)'])
df['july (avg high degree)'] = pd.to_numeric(df['july (avg high degree)'])

# Calculate average high temperatures
avg_jan_high = df['january (avg high degree)'].mean()
avg_jul_high = df['july (avg high degree)'].mean()

# Calculate the difference
difference = avg_jul_high - avg_jan_high
print(f"Final Answer: {difference:.1f}")