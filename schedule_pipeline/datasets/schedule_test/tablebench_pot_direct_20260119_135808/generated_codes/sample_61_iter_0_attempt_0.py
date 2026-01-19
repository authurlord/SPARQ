import pandas as pd

df = pd.read_csv('table.csv')
# Find the nation with Total = 57
nation_with_57_medals = df[df['Total'] == 57]['Nation'].iloc[0]
print(f"Final Answer: {nation_with_57_medals}")