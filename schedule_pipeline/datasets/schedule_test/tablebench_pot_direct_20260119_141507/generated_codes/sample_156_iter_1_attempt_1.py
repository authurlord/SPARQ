import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Remove the 'European Union' row since it's not a member state
df_filtered = df[df['member state'] != 'european union']

# Convert relevant columns to numeric (in case of formatting issues)
df_filtered['population % of eu'] = pd.to_numeric(df_filtered['population % of eu'], errors='coerce')
df_filtered['area km 2'] = pd.to_numeric(df_filtered['area km 2'], errors='coerce')
df_filtered['area % of eu'] = pd.to_numeric(df_filtered['area % of eu'], errors='coerce')
df_filtered['pop density people / km 2'] = pd.to_numeric(df_filtered['pop density people / km 2'], errors='coerce')

# Compute correlation with 'population % of eu'
correlations = df_filtered[['population % of eu', 'area km 2', 'area % of eu', 'pop density people / km 2']].corr(numeric_only=True)

# Extract absolute correlation values with 'population % of eu'
corr_with_pop = correlations['population % of eu'].abs()

# Identify factors with correlation > 0.3 (considered significant)
significant_factors = []
for factor in ['area km 2', 'area % of eu', 'pop density people / km 2']:
    if abs(corr_with_pop[factor]) > 0.3:
        significant_factors.append(factor)

if significant_factors:
    Final Answer: , ".join(significant_factors)
else:
    Final Answer: no clear impact