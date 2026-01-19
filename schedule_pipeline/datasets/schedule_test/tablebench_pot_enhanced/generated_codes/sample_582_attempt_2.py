import pandas as pd

df = pd.read_csv('table.csv')
# Convert percentage column to float
df['Percentage (based on registration by the church itself)'] = pd.to_numeric(df['Percentage (based on registration by the church itself)'])
# Calculate the year-over-year change
df['change'] = df['Percentage (based on registration by the church itself)'].diff()
# Find the row with the most negative change (largest decrease)
max_decrease_year = df.loc[df['change'].idxmin()]['year']
print(f"Final Answer: {max_decrease_year}")