import pandas as pd

df = pd.read_csv('table.csv')
# Convert percentage column to numeric
df['Percentage (based on registration by the church itself)'] = pd.to_numeric(df['Percentage (based on registration by the church itself)'])
# Calculate year-on-year change in percentage
df['change'] = df['Percentage (based on registration by the church itself)'].diff()
# Find the year with the maximum decrease (most negative change)
max_decrease_year = df.loc[df['change'].idxmin(), 'year']
print(f"Final Answer: {max_decrease_year}")