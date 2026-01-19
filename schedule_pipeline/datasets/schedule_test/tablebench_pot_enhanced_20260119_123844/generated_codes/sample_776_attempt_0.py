import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'net assets at end of year' to numeric
df['net assets at end of year'] = pd.to_numeric(df['net assets at end of year'])

# Get the last two values for trend calculation
last_two_years = df['net assets at end of year'].tail(2)
increase = last_two_years.iloc[1] - last_two_years.iloc[0]

# Project the next value
projected_value = last_two_years.iloc[1] + increase

print(f"Final Answer: {projected_value}")