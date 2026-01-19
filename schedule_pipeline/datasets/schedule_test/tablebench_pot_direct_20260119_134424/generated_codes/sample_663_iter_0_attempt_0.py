import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['commissioned capacity (mw)'] = pd.to_numeric(df['commissioned capacity (mw)'])
df['year of commission'] = pd.to_numeric(df['year of commission'])

# Calculate correlation coefficient
correlation = df['commissioned capacity (mw)'].corr(df['year of commission'])
print(f"Final Answer: {correlation:.3f}")