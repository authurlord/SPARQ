import pandas as pd

df = pd.read_csv('table.csv')
# Sort by elevation (m) in descending order and get top 3
top_3_mountains = df.sort_values(by='elevation (m)', ascending=False).head(3)[['rank', 'peak']]
# Output the rank and peak name for top 3
print(f"Final Answer: {top_3_mountains.to_string(index=False)}")