import pandas as pd

df = pd.read_csv('table.csv')

# Filter for United States rockets
us_rockets = df[df['country'] == 'united states']

# Total launches and successes for the US
total_launches_us = us_rockets['launches'].sum()
total_successes_us = us_rockets['successes'].sum()
success_rate_us = total_successes_us / total_launches_us

# Remove Falcon 9 to see the rate without it
us_without_falcon9 = us_rockets[us_rockets['rocket'] != 'falcon 9']
total_launches_without_f9 = us_without_falcon9['launches'].sum()
total_successes_without_f9 = us_without_falcon9['successes'].sum()
success_rate_without_f9 = total_successes_without_f9 / total_launches_without_f9

# Compare success rates
if success_rate_us > success_rate_without_f9:
    impact = "positive"
elif success_rate_us < success_rate_without_f9:
    impact = "negative"
else:
    impact = "no clear impact"

print(f"Final Answer: {impact}")