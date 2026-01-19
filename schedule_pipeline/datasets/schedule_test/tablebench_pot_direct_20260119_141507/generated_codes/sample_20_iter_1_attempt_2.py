import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'number of dances' to numeric, coercing errors to NaN if any
df['number of dances'] = pd.to_numeric(df['number of dances'], errors='coerce')

# Filter couples with more than 10 dances
filtered_couples = df[df['number of dances'] > 10]

# Calculate average total points for these couples
average_points = filtered_couples['total points'].mean()

print(f"Final Answer: {average_points:.1f}")