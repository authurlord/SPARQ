import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'lines' and 'annual ridership (2012)' columns
lines = df['lines']
ridership = df['annual ridership (2012)']

# Calculate the correlation coefficient
correlation = lines.corr(ridership)

# Interpret the result: positive, negative, or no clear impact
if correlation > 0.3:
    impact = "positive"
elif correlation < -0.3:
    impact = "negative"
else:
    impact = "no clear impact"

print(f"Final Answer: {impact}")