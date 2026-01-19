import pandas as pd

df = pd.read_csv('table.csv')
# Extract just (cents) and error columns
just_cents = df['just (cents)']
error = df['error']

# Calculate Pearson correlation
correlation = just_cents.corr(error)

# Print the result
print(f"Final Answer: {correlation:.3f}")