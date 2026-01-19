import pandas as pd

df = pd.read_csv('table.csv')
# Find the nation where Total is 57
nation_with_57_medals = df[df['Total'] == 57]['Nation'].values[0]
print(f"Final Answer: {nation_with_57_medals}")