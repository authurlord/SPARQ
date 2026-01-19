import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter only U.S. rockets
us_rockets = df[df['country'] == 'united states']

# Convert 'launches' and 'successes' to integers
us_rockets['launches'] = pd.to_numeric(us_rockets['launches'], errors='coerce')
us_rockets['successes'] = pd.to_numeric(us_rockets['successes'], errors='coerce')

# Remove rows with invalid values
us_rockets = us_rockets.dropna(subset=['launches', 'successes'])

# Find the index of Falcon 9
falcon9_index = us_rockets[us_rockets['rocket'] == 'falcon 9'].index[0]

# Split data into before and after Falcon 9
before_falcon9 = us_rockets.loc[:falcon9_index - 1]
after_falcon9 = us_rockets.loc[falcon9_index + 1:]

# Calculate success rates
def calculate_success_rate(group):
    if group.empty:
        return 0.0
    total_launches = group['launches'].sum()
    total_successes = group['successes'].sum()
    return total_successes / total_launches if total_launches > 0 else 0.0

pre_rate = calculate_success_rate(before_falcon9)
post_rate = calculate_success_rate(after_falcon9)

# Compare rates
if post_rate > pre_rate:
    impact = "positive"
elif post_rate < pre_rate:
    impact = "negative"
else:
    impact = "no clear impact"

print(f"Final Answer: {impact}")