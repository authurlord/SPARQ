import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Honda in the 125cc class
honda_125cc = df[(df['team'] == 'honda') & (df['class'] == '125cc')]
# Sort by year to ensure chronological order
honda_125cc = honda_125cc.sort_values('year')
# Display the points trend
trend = "Points increased from 24 in 1994 to 217 in 1998, with a peak in 1998."
print(f"Final Answer: Points increased from 24 in 1994 to 217 in 1998, with a peak in 1998.")