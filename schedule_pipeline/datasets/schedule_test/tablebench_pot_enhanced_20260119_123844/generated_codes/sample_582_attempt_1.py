import pandas as pd

df = pd.read_csv('table.csv')
# Convert percentage column to float
df['Percentage (based on registration by the church itself)'] = pd.to_numeric(df['Percentage (based on registration by the church itself)'])
# Calculate the year-on-year change in percentage
df['percentage_change'] = df['Percentage (based on registration by the church itself)'].diff()
# Find the row with the minimum (most negative) change
min_change_year = df.loc[df['percentage_change'].idxmin()]['year']
print(f"Final Answer: {min_change_year}")