import pandas as pd

df = pd.read_csv('table.csv')
# Convert percentage column to float
df['Percentage (based on registration by the church itself)'] = pd.to_numeric(df['Percentage (based on registration by the church itself)'])
# Calculate year-on-year change
df['change'] = df['Percentage (based on registration by the church itself)'].diff()
# Find the year with the maximum decrease (most negative change)
min_change_year = df.loc[df['change'].idxmin(), 'year']
print(f"Final Answer: {min_change_year}")