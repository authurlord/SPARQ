import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'typhus' column to numeric
df['typhus'] = pd.to_numeric(df['typhus'])
# Calculate the year-on-year change in typhus cases
df['typhus_change'] = df['typhus'].diff()
# Find the row with the most negative change (largest decrease)
max_decrease_year = df.loc[df['typhus_change'].idxmin()]['year']
print(f"Final Answer: {max_decrease_year}")