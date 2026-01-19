import pandas as pd

df = pd.read_csv('table.csv')

# Convert all values in the numeric columns to float
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Identify outliers using IQR method for each column
outliers = []
for col in df.columns[1:]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Find values outside the bounds
    outliers_in_col = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    for _, row in outliers_in_col.iterrows():
        outliers.append({
            'term': row['term abroad'],
            'year': row.name,
            'value': row[col],
            'column': col
        })

# Print the list of outliers
if outliers:
    for out in outliers:
        print(f"Outlier: {out['term']} in {out['year']} ({out['column']}) = {out['value']}")
else:
    print("No significant deviations found.")