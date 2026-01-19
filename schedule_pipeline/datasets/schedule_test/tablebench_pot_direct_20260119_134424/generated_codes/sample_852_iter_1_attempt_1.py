import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'average' column to float
df['average'] = pd.to_numeric(df['average'], errors='coerce')
# Get average score of top-ranked and bottom-ranked couples
top_average = df.iloc[0]['average']
bottom_average = df.iloc[-1]['average']
# Calculate the difference
difference = top_average - bottom_average
print(f"Final Answer: {difference:.1f}")