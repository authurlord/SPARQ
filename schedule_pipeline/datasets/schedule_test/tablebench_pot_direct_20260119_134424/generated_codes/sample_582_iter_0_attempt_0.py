import pandas as pd

df = pd.read_csv('table.csv')
# Convert percentage column to float
df['Percentage (based on registration by the church itself)'] = df['Percentage (based on registration by the church itself)'].astype(float)
# Calculate year-over-year change
df['change'] = df['Percentage (based on registration by the church itself)'].diff()
# Find the row with the most negative change (largest decrease)
most_decrease_year = df.loc[df['change'].idxmin()]['year']
print(f"Final Answer: {most_decrease_year}")