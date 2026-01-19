import pandas as pd

df = pd.read_csv('table.csv')
# Sort by elevation (m) in descending order and get top 3
top_3_mountains = df.sort_values(by='elevation (m)', ascending=False).head(3)
# Extract rank and peak name
result = [f"{row['rank']} - {row['peak']}" for _, row in top_3_mountains.iterrows()]
print(f"Final Answer: {', '.join(result)}")