import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'lines' and 'annual ridership (2012)'
correlation = df['lines'].corr(df['annual ridership (2012)'])
print(f"Final Answer: {'positive' if correlation > 0 else 'negative' if correlation < 0 else 'no clear'}")