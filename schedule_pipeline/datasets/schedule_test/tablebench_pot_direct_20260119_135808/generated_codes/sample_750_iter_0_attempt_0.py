import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where starts = 3 (same as 2004)
three_starts_data = df[df['starts'] == 3]

# Get the avg finish for those seasons
forecasted_avg_finish = three_starts_data['avg finish'].mean()

print(f"Final Answer: {forecasted_avg_finish:.1f}")