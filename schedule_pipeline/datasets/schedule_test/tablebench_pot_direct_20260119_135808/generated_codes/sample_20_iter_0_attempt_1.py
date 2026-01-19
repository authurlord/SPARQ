import pandas as pd

df = pd.read_csv('table.csv')
# Filter couples who performed more than 10 dances
filtered_df = df[df['number of dances'] > 10]
# Calculate average total points
avg_points = filtered_df['total points'].mean()
print(f"Final Answer: {avg_points:.1f}")