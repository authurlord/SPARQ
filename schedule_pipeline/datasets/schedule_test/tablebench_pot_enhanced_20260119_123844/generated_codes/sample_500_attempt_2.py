import pandas as pd

df = pd.read_csv('table.csv')
# Sort by Death toll in descending order and take top 5
top_5_by_death_toll = df.sort_values(by='Death toll', ascending=False).head(5)
# Calculate average magnitude
avg_magnitude = top_5_by_death_toll['Magnitude'].mean()
print(f"Final Answer: {avg_magnitude:.1f}")