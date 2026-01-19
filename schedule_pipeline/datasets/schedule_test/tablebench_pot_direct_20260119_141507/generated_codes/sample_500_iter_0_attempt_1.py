import pandas as pd

df = pd.read_csv('table.csv')
# Sort by 'Death toll' in descending order and take top 5
top_5_by_death_toll = df.sort_values(by='Death toll', ascending=False).head(5)
# Convert 'Magnitude' to float and calculate the mean
average_magnitude = top_5_by_death_toll['Magnitude'].astype(float).mean()
print(f"Final Answer: {average_magnitude:.1f}")