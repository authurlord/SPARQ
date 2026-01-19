import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'number of dances' to integer for proper comparison
df['number of dances'] = pd.to_numeric(df['number of dances'])
# Filter couples who performed more than 10 dances
filtered_df = df[df['number of dances'] > 10]
# Calculate average total points
average_points = filtered_df['total points'].astype(int).mean()
print(f"Final Answer: {average_points:.1f}")