import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert registration columns to integers
for col in ['2002 registrations', '2005 registrations', '2011 registrations']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Function to find outliers per year using 10th and 90th percentiles
def find_outliers_by_year(year_col):
    q1 = df[year_col].quantile(0.10)
    q3 = df[year_col].quantile(0.90)
    lower_bound = q1 * 0.95
    upper_bound = q3 * 1.05
    outliers = df[(df[year_col] < lower_bound) | (df[year_col] > upper_bound)]
    return outliers[['breed', 'group', year_col]]

# Find outliers for each year
outliers_2002 = find_outliers_by_year('2002 registrations')
outliers_2005 = find_outliers_by_year('2005 registrations')
outliers_2011 = find_outliers_by_year('2011 registrations')

# Combine all outlier entries
all_outliers = pd.concat([outliers_2002, outliers_2005, outliers_2011], ignore_index=True)

# Get unique breeds with outliers
unique_outlier_breeds = all_outliers['breed'].dropna().unique()

# Final answer: list of breeds that have unusually high or low registrations in any year
Final Answer: bloodhound, english setter, gordon setter, spaniel (clumber), soft coated wheaten terrier, welsh corgi (cardigan), welsh terrier