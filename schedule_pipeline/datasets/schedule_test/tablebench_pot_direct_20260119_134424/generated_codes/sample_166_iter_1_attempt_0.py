import pandas as pd

df = pd.read_csv('table.csv')

# Filter for rockets from the United States
us_rockets = df[df['country'] == 'united states']

# Find the index of the first Falcon 9 launch
falcon9_index = us_rockets[us_rockets['rocket'] == 'falcon 9'].index[0]

# Split into two groups: before and after Falcon 9
before_falcon9 = us_rockets.iloc[:falcon9_index]
after_falcon9 = us_rockets.iloc[falcon9_index:]

# Calculate success rates
def calculate_success_rate(group):
    total = group['launches'].sum()
    successes = group['successes'].sum()
    return successes / total if total > 0 else 0

before_success_rate = calculate_success_rate(before_falcon9)
after_success_rate = calculate_success_rate(after_falcon9)

# Compare success rates
if after_success_rate > before_success_rate:
    impact = "positive"
elif after_success_rate < before_success_rate:
    impact = "negative"
else:
    impact = "no clear impact"

print(f"Final Answer: {impact}")