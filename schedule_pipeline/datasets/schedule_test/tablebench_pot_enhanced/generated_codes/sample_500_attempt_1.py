import pandas as pd

df = pd.read_csv('table.csv')
# Sort by Death toll in descending order
df_sorted = df.sort_values(by='Death toll', ascending=False)
# Get the top 5 earthquakes
top_5 = df_sorted.head(5)
# Calculate average magnitude
avg_magnitude = top_5['Magnitude'].mean()
print(f"Final Answer: {avg_magnitude:.1f}")