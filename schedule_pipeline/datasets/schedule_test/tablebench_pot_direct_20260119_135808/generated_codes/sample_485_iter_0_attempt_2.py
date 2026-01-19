import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Top speed' to numeric for analysis
df['Top speed'] = pd.to_numeric(df['Top speed'], errors='coerce')

# Identify outliers in 'Top speed' using IQR method
Q1 = df['Top speed'].quantile(0.25)
Q3 = df['Top speed'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_speed = df[(df['Top speed'] < lower_bound) | (df['Top speed'] > upper_bound)]

# Check for unusual 'Built' dates (e.g., future years)
df['Built'] = df['Built'].astype(str)
future_years = df[df['Built'].str.contains('20[5-9][0-9]|21[0-9][0-9]', na=False)]

# Combine findings
outliers = outliers_speed['Class'].tolist()
if not outliers:
    outliers = []

# Add class with future built date if any
if not future_years.empty:
    future_classes = future_years['Class'].tolist()
    outliers.extend(future_classes)

# Remove duplicates and ensure output is clean
outliers = list(set(outliers))

print(f"Final Answer: {', '.join(outliers)}")