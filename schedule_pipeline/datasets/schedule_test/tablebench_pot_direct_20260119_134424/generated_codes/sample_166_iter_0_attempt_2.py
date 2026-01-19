import pandas as pd

df = pd.read_csv('table.csv')

# Filter for United States rockets
us_rockets = df[df['country'] == 'united states']

# Identify the Falcon 9 rocket
falcon9_row = df[df['rocket'] == 'falcon 9']
falcon9_launches = falcon9_row['launches'].values[0]

# Get the total launches and successes before Falcon 9
before_falcon9 = us_rockets[us_rockets['rocket'] != 'falcon 9']
before_success_rate = before_falcon9['successes'].sum() / before_falcon9['launches'].sum()

# Include Falcon 9 in the after group
after_falcon9 = us_rockets[us_rockets['rocket'] == 'falcon 9']
after_success_rate = after_falcon9['successes'].sum() / after_falcon9['launches'].sum()

# Overall success rate including Falcon 9
overall_success_rate = us_rockets['successes'].sum() / us_rockets['launches'].sum()

# Compare success rates
if after_success_rate > before_success_rate:
    impact = "positive"
elif after_success_rate < before_success_rate:
    impact = "negative"
else:
    impact = "no clear impact"

print(f"Final Answer: {impact}")