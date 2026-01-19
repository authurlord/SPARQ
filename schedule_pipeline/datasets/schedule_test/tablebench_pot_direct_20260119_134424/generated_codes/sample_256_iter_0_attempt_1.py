import pandas as pd

df = pd.read_csv('table.csv')

# Extract the top 3 countries by Primary schools
top_3_primary = df.nlargest(3, 'Primary')['Primary'].astype(int).sum()

# Total Career-related schools from the "Total Schools Globally" row
total_career_related = int(df.loc[df['Country'] == 'Total Schools Globally', 'Career-related'].values[0])

# Calculate difference
difference = top_3_primary - total_career_related

print(f"Final Answer: {difference}")