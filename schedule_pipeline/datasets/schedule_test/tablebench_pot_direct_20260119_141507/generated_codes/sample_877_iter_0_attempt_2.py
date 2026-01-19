import pandas as pd

df = pd.read_csv('table.csv')
# Convert average high temperatures to numeric
jan_high = df['january (avg high degree)'].astype(float)
jul_high = df['july (avg high degree)'].astype(float)

# Calculate the average high temperatures for January and July
avg_jan_high = jan_high.mean()
avg_jul_high = jul_high.mean()

# Find the difference
temp_difference = avg_jul_high - avg_jan_high
print(f"Final Answer: {temp_difference:.1f}")