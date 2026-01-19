import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'typhus' column to numeric
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')

# Calculate the difference in typhus cases from the previous year
df['typhus_diff'] = df['typhus'].diff()

# Find the row with the most negative difference (largest decrease)
min_decrease_year = df.loc[df['typhus_diff'].idxmin()]['year']

print(f"Final Answer: {min_decrease_year}")