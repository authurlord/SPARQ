import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter rows where City is 'Moscow'
moscow_data = df[df['City'] == 'Moscow']

# Convert Year to integer and drop rows with missing or invalid data
moscow_data['Year'] = pd.to_numeric(moscow_data['Year'], errors='coerce')
moscow_data = moscow_data.dropna(subset=['Year'])

# Sort by Year
moscow_data = moscow_data.sort_values('Year')

# Extract total examinees
totals = moscow_data['Number of Examinees by Level_4'].astype(float).fillna(0)

# Compute year-over-year percentage change
percentage_changes = []
for i in range(1, len(totals)):
    prev = totals.iloc[i-1]
    curr = totals.iloc[i]
    if prev == 0:
        continue
    change = ((curr - prev) / prev) * 100
    percentage_changes.append(change)

# Find the year with the maximum percentage increase
max_increase_idx = percentage_changes.index(max(percentage_changes)) if percentage_changes else None
if max_increase_idx is not None:
    max_year = moscow_data['Year'].iloc[max_increase_idx + 1]
else:
    max_year = None

print(f"Final Answer: {max_year}")