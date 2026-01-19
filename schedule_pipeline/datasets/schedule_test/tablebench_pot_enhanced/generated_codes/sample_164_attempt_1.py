import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric for analysis
df['Labour councillors'] = pd.to_numeric(df['Labour councillors'])
df['Conservative councillors'] = pd.to_numeric(df['Conservative councillors'])
df['Independent councillors'] = pd.to_numeric(df['Independent councillors'])
df['Liberal councillors'] = pd.to_numeric(df['Liberal councillors'])

# Compute correlation between Labour councillors and other groups
correlations = df[['Labour councillors', 'Conservative councillors', 'Independent councillors', 'Liberal councillors']].corr()['Labour councillors'].drop('Labour councillors')

# Check if any correlation is strong (absolute value > 0.5)
strong_correlation = correlations[abs(correlations) > 0.5]

if len(strong_correlation) == 0:
    print("Final Answer: no clear impact")
else:
    # List the influencing factors
    influencing_factors = ', '.join(strong_correlation.index)
    print(f"Final Answer: {influencing_factors}")