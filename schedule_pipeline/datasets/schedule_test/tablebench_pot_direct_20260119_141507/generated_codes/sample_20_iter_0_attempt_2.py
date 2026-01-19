import pandas as pd

df = pd.read_csv('table.csv')
# Filter couples with more than 10 dances
filtered_couples = df[df['number of dances'] > 10]
# Calculate average total points for these couples
average_points = filtered_couples['total points'].mean()
print(f"Final Answer: {average_points:.1f}")