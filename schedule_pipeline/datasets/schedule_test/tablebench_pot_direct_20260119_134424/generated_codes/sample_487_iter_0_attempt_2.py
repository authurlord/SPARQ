import pandas as pd

df = pd.read_csv('table.csv')

# Convert all columns except 'term abroad' to numeric
numeric_columns = df.columns[1:]
df[numeric_columns] = df[numeric_columns].astype(float)

# Initialize list to store outliers
outliers = []

# Iterate over each row (term abroad)
for _, row in df.iterrows():
    term_abroad = row['term abroad']
    values = row[numeric_columns]
    mean_val = values.mean()
    std_val = values.std()
    
    # Identify values with Z-score > 2
    z_scores = (values - mean_val) / std_val
    for col, z_score in z_scores.items():
        if abs(z_score) > 2:
            outliers.append((term_abroad, col, values[col]))

# Print the outliers
print(f"Final Answer: {outliers}")