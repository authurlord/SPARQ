import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for Honda in 125cc class
filtered_df = df[(df['team'] == 'honda') & (df['class'] == '125cc')]

# Extract points and years
points = filtered_df['points'].astype(int)
years = pd.to_numeric(filtered_df['year'])

# Compute annual increase between consecutive years
differences = []
for i in range(1, len(years)):
    diff = points.iloc[i] - points.iloc[i-1]
    differences.append(diff)

# Calculate average annual increase
avg_increase = sum(differences) / len(differences) if differences else 0
print(f"Final Answer: {avg_increase:.1f}")