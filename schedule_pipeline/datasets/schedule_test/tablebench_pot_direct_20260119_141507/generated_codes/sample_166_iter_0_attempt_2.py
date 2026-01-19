import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter U.S. rockets
us_rockets = df[df['country'] == 'united states']

# Calculate total success rate for all U.S. rockets
total_launches_us = us_rockets['launches'].sum()
total_successes_us = us_rockets['successes'].sum()
overall_success_rate_us = total_successes_us / total_launches_us if total_launches_us > 0 else 0

# Exclude Falcon 9 to get pre-Falcon 9 success rate
us_rockets_without_falcon9 = us_rockets[us_rockets['rocket'] != 'falcon 9']
pre_falcon9_launches = us_rockets_without_falcon9['launches'].sum()
pre_falcon9_successes = us_rockets_without_falcon9['successes'].sum()
pre_falcon9_success_rate = pre_falcon9_successes / pre_falcon9_launches if pre_falcon9_launches > 0 else 0

# Compare success rates
if overall_success_rate_us > pre_falcon9_success_rate:
    impact = "positive"
elif overall_success_rate_us < pre_falcon9_success_rate:
    impact = "negative"
else:
    impact = "no clear impact"

print(f"Final Answer: {impact}")