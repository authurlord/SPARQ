import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'commissioned capacity (mw)' and 'year of commission'
correlation = df['commissioned capacity (mw)'].corr(df['year of commission'])
print(f"Final Answer: {correlation:.2f}")