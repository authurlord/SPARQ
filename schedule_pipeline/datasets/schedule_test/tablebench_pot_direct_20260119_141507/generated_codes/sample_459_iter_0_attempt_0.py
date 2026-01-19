import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert all values in the columns (excluding 'county') to numeric
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN (due to conversion errors)
df = df.dropna()

# Extract the county names and the data
counties = df['county'].tolist()
data = df.iloc[:, 1:].values.astype(float)

# Define a threshold for unusual change (e.g., >20%)
threshold = 0.20

# List to store counties with unusual patterns
unusual_counties = []

# Check year-over-year changes between consecutive years
years = df.columns[1:]
for i in range(len(years) - 1):
    year1, year2 = years[i], years[i+1]
    # Calculate percentage change
    changes = (data[:, i] - data[:, i+1]) / data[:, i+1] * 100
    # Find counties where change exceeds threshold
    for j, change in enumerate(changes):
        if abs(change) > threshold:
            county = counties[j]
            if county not in unusual_counties:
                unusual_counties.append(county)

# Final answer: list of counties with unusual patterns
print(f"Final Answer: {', '.join(unusual_counties)}")