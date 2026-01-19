import pandas as pd

df = pd.read_csv('table.csv')
# Clean 'US Chart position' column: extract only the numeric part
df['US Chart position'] = df['US Chart position'].astype(str).str.extract('(\d+)').astype(float)
# Calculate the average
avg_chart_position = df['US Chart position'].mean()
print(f"Final Answer: {avg_chart_position:.1f}")