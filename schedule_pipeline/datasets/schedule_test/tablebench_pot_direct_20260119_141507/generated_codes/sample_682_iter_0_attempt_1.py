import pandas as pd

df = pd.read_csv('table.csv')
# Convert string values to float for numerical computation
df_numeric = df[['1990 - 95', '2006 - 10']].apply(pd.to_numeric, errors='coerce')
# Calculate the correlation coefficient
correlation = df_numeric['1990 - 95'].corr(df_numeric['2006 - 10'])
print(f"Final Answer: {correlation:.3f}")