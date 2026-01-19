import pandas as pd

df = pd.read_csv('table.csv')
# Calculate medal ratio: gold / total medals
df['medal_ratio'] = df['gold medals'].astype(float) / df['total medals'].astype(float)
# Find the school with the highest ratio
highest_ratio_school = df.loc[df['medal_ratio'].idxmax(), 'ensemble']
print(f"Final Answer: {highest_ratio_school}")