import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter U.S. rockets
us_rockets = df[df['country'] == 'united states']

# Calculate total launches and successes for U.S. rockets (excluding Falcon 9)
us_rockets_excluding_falcon9 = us_rockets[us_rockets['rocket'] != 'falcon 9']

# Total launches and successes for U.S. rockets excluding Falcon 9
total_launches_excl = us_rockets_excluding_falcon9['launches'].sum()
total_successes_excl = us_rockets_excluding_falcon9['successes'].sum()

# Success rate excluding Falcon 9
success_rate_excl = total_successes_excl / total_launches_excl if total_launches_excl > 0 else 0

# Falcon 9 performance
falcon9_launches = us_rockets[us_rockets['rocket'] == 'falcon 9']['launches'].sum()
falcon9_successes = us_rockets[us_rockets['rocket'] == 'falcon 9']['successes'].sum()
falcon9_rate = falcon9_successes / falcon9_launches if falcon9_launches > 0 else 0

# Overall U.S. success rate including Falcon 9
total_us_launches = us_rockets['launches'].sum()
total_us_successes = us_rockets['successes'].sum()
overall_us_rate = total_us_successes / total_us_launches if total_us_launches > 0 else 0

# Compare success rates
if falcon9_rate > success_rate_excl:
    impact = "positive"
elif falcon9_rate < success_rate_excl:
    impact = "negative"
else:
    impact = "no clear impact"

print(f"Final Answer: {impact}")