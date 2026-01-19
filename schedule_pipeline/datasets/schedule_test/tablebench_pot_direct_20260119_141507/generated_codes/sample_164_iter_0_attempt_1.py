import pandas as pd

df = pd.read_csv('table.csv')

# Drop rows with missing values (though none present)
# Compute correlation matrix with Labour councillors
correlation = df.corr(numeric_only=True)

# Extract correlation with Labour councillors
labour_corr = correlation['Labour councillors']

# Find the strongest non-trivial correlation (excluding itself)
labour_corr_values = labour_corr.drop('Labour councillors')
strongest_corr = labour_corr_values.abs().idxmax()

# If the strongest correlation is below a threshold (say 0.3), then no clear impact
threshold = 0.3
if labour_corr_values.abs().max() < threshold:
    print("Final Answer: no clear impact")
else:
    print(f"Final Answer: {strongest_corr}")