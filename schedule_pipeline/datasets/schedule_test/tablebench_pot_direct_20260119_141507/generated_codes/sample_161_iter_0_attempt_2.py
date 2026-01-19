import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'share' from string with '%' to float
df['share'] = df['share'].str.replace('%', '').astype(float)

# Calculate correlation with 'bbc one weekly ranking'
correlations = df[['total viewers', 'share']].corrwith(df['bbc one weekly ranking'])

# Check if any correlation has absolute value > 0.3 (considered significant)
significant_factors = []
for col in ['total viewers', 'share']:
    if abs(correlations[col]) > 0.3:
        significant_factors.append(col)

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")