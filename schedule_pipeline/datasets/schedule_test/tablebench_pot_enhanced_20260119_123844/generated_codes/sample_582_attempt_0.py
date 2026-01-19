import pandas as pd

df = pd.read_csv('table.csv')
# Convert percentage column to float
df['Percentage (based on registration by the church itself)'] = pd.to_numeric(df['Percentage (based on registration by the church itself)'])
# Calculate the difference in percentage from the previous year
df['change'] = df['Percentage (based on registration by the church itself)'].diff()
# Find the year with the most negative change (largest decrease)
worst_year = df.loc[df['change'].idxmin(), 'year']
print(f"Final Answer: {worst_year}")