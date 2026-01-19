import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Honda in 125cc class
honda_125cc = df[(df['team'] == 'honda') & (df['class'] == '125cc')]

# Extract year and points for trend observation
trend_data = honda_125cc[['year', 'points']].to_dict('records')
print(f"Final Answer: The points scored by Honda in the 125cc class increased from 24 in 1994 to 217 in 1998.")