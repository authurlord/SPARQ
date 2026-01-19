import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert string values to numeric where possible (e.g., 'n/a' to NaN)
df = df.replace('n / a', pd.NA)

# Select relevant columns
columns_to_analyze = [
    'tourist arrivals (2011) (millions)',
    'tourism receipts (2011) (millions of us)',
    'tourism receipts (2011) (us per arrival)',
    'tourism receipts (2011) (us per capita)',
    'tourism receipts (2003) (as % of gdp)',
    'tourism competitiveness (2011) (ttci)'
]

# Convert to numeric, handling errors
for col in columns_to_analyze:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Compute correlation with 'tourism receipts (2011) (millions of us)'
correlation_with_receipts = df[columns_to_analyze].corrwith(df['tourism receipts (2011) (millions of us)'])

# Filter only those with absolute correlation > 0.5
significant_correlations = correlation_with_receipts[abs(correlation_with_receipts) > 0.5]

# If no significant correlation, return "no clear impact"
if significant_correlations.empty:
    print("Final Answer: no clear impact")
else:
    # Return the names of the significant factors
    significant_factors = list(significant_correlations.index)
    print(f"Final Answer: {', '.join(significant_factors)}")