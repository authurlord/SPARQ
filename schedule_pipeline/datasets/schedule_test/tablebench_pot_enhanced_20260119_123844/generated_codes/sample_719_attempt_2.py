import pandas as pd

df = pd.read_csv('table.csv')
# Sort by elevation in descending order and get top 3
top_3_mountains = df.sort_values(by='elevation (m)', ascending=False).head(3)['peak'].tolist()
print(f"Final Answer: {', '.join(top_3_mountains)}")