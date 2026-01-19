import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where starts = 3 (same as 2004)
filtered_data = df[df['starts'] == 3]
# Get the avg finish for those rows
forecasted_avg_finish = filtered_data['avg finish'].iloc[0]  # Only one row with 3 starts (2004)
print(f"Final Answer: {forecasted_avg_finish}")