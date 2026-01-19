import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 1985 to 2004 and where starts = 3 (same as 2004)
filtered_data = df[(df['starts'] == 3) & (df['year'].astype(int) <= 2004)]
# Calculate the average of avg finish for those rows
forecasted_avg_finish = filtered_data['avg finish'].mean()
print(f"Final Answer: {forecasted_avg_finish:.1f}")