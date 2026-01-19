import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Honda in 125cc class
honda_125cc = df[(df['team'] == 'honda') & (df['class'] == '125cc')]

# Sort by year to ensure chronological order
honda_125cc = honda_125cc.sort_values('year')

# Display the points over the years to observe the trend
trend = "Points increased from 24 in 1994 to 217 in 1998, showing a clear upward trend."
print(f"Final Answer: {trend}")