import pandas as pd

df = pd.read_csv('table.csv')
# Convert relevant columns to numeric
df['pleasure'] = pd.to_numeric(df['pleasure'])
df['psychological dependence'] = pd.to_numeric(df['psychological dependence'])
df['physical dependence'] = pd.to_numeric(df['physical dependence'])

# Calculate correlations
corr_pleasure_psych = df['pleasure'].corr(df['psychological dependence'])
corr_pleasure_phys = df['pleasure'].corr(df['physical dependence'])

# Determine which correlation is stronger
if corr_pleasure_psych > corr_pleasure_phys:
    result = "psychological dependence"
else:
    result = "physical dependence"

print(f"Final Answer: {result}")