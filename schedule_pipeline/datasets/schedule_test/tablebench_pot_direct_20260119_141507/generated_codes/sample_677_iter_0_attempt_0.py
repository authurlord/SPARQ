import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'total renewable generation' and 'total electricity demand'
correlation = df['total renewable generation'].corr(df['total electricity demand'])
print(f"Final Answer: {correlation:.3f}")