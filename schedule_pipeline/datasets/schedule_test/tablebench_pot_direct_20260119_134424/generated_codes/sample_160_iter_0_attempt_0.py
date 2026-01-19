import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric, handling missing values
df['Outright Titles'] = pd.to_numeric(df['Outright Titles'], errors='coerce')
df['Shared Titles'] = pd.to_numeric(df['Shared Titles'], errors='coerce')
df['Runners-Up'] = pd.to_numeric(df['Runners-Up'], errors='coerce')
df['Total Finals'] = pd.to_numeric(df['Total Finals'], errors='coerce')

# Drop rows with missing values in key columns
df_clean = df.dropna(subset=['Outright Titles', 'Shared Titles', 'Runners-Up', 'Total Finals'])

# Calculate correlation with 'Outright Titles'
correlations = df_clean[['Outright Titles', 'Shared Titles', 'Runners-Up', 'Total Finals']].corr()['Outright Titles'].abs()

# Check if any correlation is above 0.7
strong_impact = correlations[correlations > 0.7].index.tolist()

if len(strong_impact) == 0:
    print("Final Answer: no clear impact")
else:
    # Exclude 'Outright Titles' itself from the result
    impact_factors = [factor for factor in strong_impact if factor != 'Outright Titles']
    print(f"Final Answer: {', '.join(impact_factors)}")