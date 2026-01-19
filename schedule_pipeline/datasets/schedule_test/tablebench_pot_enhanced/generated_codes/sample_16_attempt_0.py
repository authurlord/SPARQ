import pandas as pd

df = pd.read_csv('table.csv')
# Clean the 'US Chart position' column by extracting numeric values
df['US Chart position'] = df['US Chart position'].astype(str).str.extract('(\d+)').astype(float)
# Calculate the average
avg_chart_position = df['US Chart position'].mean()
print(f"Final Answer: {avg_chart_position:.1f}")