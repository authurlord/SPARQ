import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Honda in the 125cc class
honda_125cc = df[(df['team'] == 'honda') & (df['class'] == '125cc')]
# Sort by year to ensure chronological order
honda_125cc = honda_125cc.sort_values('year')
# Display the points trend
trend = "increasing" if honda_125cc['points'].diff().dropna().gt(0).all() else "not consistently increasing"
print(f"Final Answer: {trend}")