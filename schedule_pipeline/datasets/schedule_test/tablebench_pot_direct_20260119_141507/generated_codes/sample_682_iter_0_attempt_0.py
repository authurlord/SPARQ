import pandas as pd

df = pd.read_csv('table.csv')
# Convert string values to float
df[['1990 - 95', '2006 - 10']] = df[['1990 - 95', '2006 - 10']].apply(pd.to_numeric, errors='coerce')

# Calculate the correlation coefficient between '1990 - 95' and '2006 - 10'
correlation = df['1990 - 95'].corr(df['2006 - 10'])
print(f"Final Answer: {correlation:.3f}")