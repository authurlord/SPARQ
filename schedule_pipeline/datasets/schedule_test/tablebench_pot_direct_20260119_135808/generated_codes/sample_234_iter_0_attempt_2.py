import pandas as pd

df = pd.read_csv('table.csv')

# Convert columns to numeric
df['pleasure'] = pd.to_numeric(df['pleasure'])
df['psychological dependence'] = pd.to_numeric(df['psychological dependence'])
df['physical dependence'] = pd.to_numeric(df['physical dependence'])

# Calculate correlation coefficients
corr_psych = df['pleasure'].corr(df['psychological dependence'])
corr_phys = df['pleasure'].corr(df['physical dependence'])

# Determine which correlation is stronger
if abs(corr_psych) > abs(corr_phys):
    result = "psychological dependence"
else:
    result = "physical dependence"

print(f"Final Answer: {result}")