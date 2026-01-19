import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric
df['judges'] = pd.to_numeric(df['judges'])
df['public'] = pd.to_numeric(df['public'])
df['total'] = pd.to_numeric(df['total'])
df['vote percentage'] = df['vote percentage'].str.replace('%', '').astype(float)

# Compute correlation between each factor and total score
correlations = df[['judges', 'public', 'vote percentage', 'total']].corr()['total'].drop('total')

# Check if any correlation is strong (absolute value > 0.5)
strong_contributors = [col for col, corr in correlations.items() if abs(corr) > 0.5]

if len(strong_contributors) > 0:
    print(f"Final Answer: {', '.join(strong_contributors)}")
else:
    print("Final Answer: no clear impact")