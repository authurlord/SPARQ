import pandas as pd

df = pd.read_csv('table.csv')
# Convert the 'Percentage' column to numeric
df['Percentage'] = pd.to_numeric(df['Percentage'], errors='coerce')

# Calculate the difference in percentage from year to year
df['diff'] = df['Percentage'].diff()

# Find the year with the maximum negative difference (largest decrease)
decrease_year = df[df['diff'] == df['diff'].min()]['year'].values[0]

print(f"Final Answer: {decrease_year}")