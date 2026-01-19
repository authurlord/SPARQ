import pandas as pd

df = pd.read_csv('table.csv')

# Convert annual ridership from string to numeric (remove commas)
df['annual ridership (2012)'] = df['annual ridership (2012)'].str.replace(',', '').astype(int)

# Calculate correlation between 'lines' and 'annual ridership (2012)'
correlation = df['lines'].corr(df['annual ridership (2012)'])

# Interpret the correlation: positive, negative, or no clear impact
if correlation > 0.3:
    impact = "positive"
elif correlation < -0.3:
    impact = "negative"
else:
    impact = "no clear impact"

print(f"Final Answer: {impact}")