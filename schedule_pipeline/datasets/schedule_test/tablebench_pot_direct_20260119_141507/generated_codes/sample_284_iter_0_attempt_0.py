import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where issue price is 34.95
filtered_df = df[df['issue price'] == '34.95'].copy()
# Sort by year to ensure chronological order
filtered_df = filtered_df.sort_values('year').reset_index(drop=True)
# Extract total mintage values
mintage_values = filtered_df['total mintage'].astype(int)
# Compute percentage increase between consecutive years
if len(mintage_values) <= 1:
    # If less than 2 years, no percentage increase can be calculated
    percentage_increases = [0]
else:
    # Calculate percentage increase from year to year
    increases = []
    for i in range(1, len(mintage_values)):
        prev = mintage_values[i-1]
        curr = mintage_values[i]
        increase = ((curr - prev) / prev) * 100
        increases.append(increase)
    percentage_increases = increases

# Average of the percentage increases
avg_increase = sum(percentage_increases) / len(percentage_increases) if percentage_increases else 0
print(f"Final Answer: {avg_increase:.1f}")