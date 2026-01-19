import pandas as pd
import re

df = pd.read_csv('table.csv')

# Function to extract year from whenbuilt
def extract_year(whenbuilt):
    if pd.isna(whenbuilt):
        return None
    if isinstance(whenbuilt, str):
        # Match patterns like '1948', 'march 1950', 'april 1950', etc.
        match = re.search(r'(\d{4})', whenbuilt)
        if match:
            return int(match.group(1))
        # Check for known non-year strings
        if whenbuilt == 'abc':
            return None
        if whenbuilt == '1234':
            return 1234
    return None

# Apply the function
df['year'] = df['whenbuilt'].apply(extract_year)

# Identify outliers: values that are extremely early or invalid
outliers = df[(df['year'] == 1234) | (df['year'].isna()) | (df['year'] < 1940)]

# Extract the original name and whenbuilt for these rows
anomalous_entries = []
for _, row in outliers.iterrows():
    anomalous_entries.append({
        'name': row['name'],
        'whenbuilt': row['whenbuilt'],
        'year': row['year']
    })

print(f"Final Answer: {anomalous_entries}")