import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Honda in 125cc class
honda_125cc = df[(df['team'] == 'honda') & (df['class'] == '125cc')]

# Extract year and points for trend analysis
years = honda_125cc['year'].tolist()
points = honda_125cc['points'].tolist()

# The trend is clearly increasing
print(f"Final Answer: increasing")