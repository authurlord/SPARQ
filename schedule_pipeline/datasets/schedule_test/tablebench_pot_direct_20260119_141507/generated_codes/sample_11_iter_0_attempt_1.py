import pandas as pd

df = pd.read_csv('table.csv')
# Sort by height (m) in descending order and take top 5
top_5 = df.sort_values(by='height (m)', ascending=False).head(5)
# Calculate the average height of these 5 buildings
avg_height = top_5['height (m)'].mean()
print(f"Final Answer: {avg_height:.1f}")