import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'total flights' and 'max altitude (miles)'
correlation = df['total flights'].corr(df['max altitude (miles)'])
print(f"Final Answer: no")