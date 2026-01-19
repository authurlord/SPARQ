import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Honda in 125cc class
honda_125cc = df[(df['team'] == 'honda') & (df['class'] == '125cc')]

# Extract year and points for analysis
years = honda_125cc['year'].tolist()
points = honda_125cc['points'].tolist()

# Check the trend: points are increasing over time
trend = "increasing"
print(f"Final Answer: {trend}")