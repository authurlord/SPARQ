import pandas as pd

df = pd.read_csv('table.csv')
# Filter for banking industry companies
banking_companies = df[df['industry'] == 'banking']

# Check if any banking companies exist
if banking_companies.empty:
    print("Final Answer: 0")
else:
    # Compute correlation between assets and profits for banking companies
    correlation = banking_companies[['assets (billion)', 'profits (billion)']].corr().iloc[0, 1]
    print(f"Final Answer: {correlation:.3f}")