import pandas as pd

df = pd.read_csv('table.csv')

# Filter the data for years with 3 starts (like 2004)
recent_data = df[df['starts'] == 3]

# Get the avg finish for the year with 3 starts
forecasted_avg_finish = recent_data['avg finish'].iloc[0]

print(f"Final Answer: {forecasted_avg_finish}")