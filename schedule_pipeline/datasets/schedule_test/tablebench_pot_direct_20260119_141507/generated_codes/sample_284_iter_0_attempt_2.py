import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where issue price is $34.95
filtered_df = df[df['issue price'] == '34.95'].copy()

# Convert 'total mintage' to numeric and sort by year
filtered_df['total mintage'] = pd.to_numeric(filtered_df['total mintage'], errors='coerce')
filtered_df = filtered_df.sort_values('year').dropna()

# If less than 2 rows, no percentage increase can be calculated
if len(filtered_df) < 2:
    print("Final Answer: 0.0")
else:
    # Calculate percentage increase from one year to the next
    mintage_series = filtered_df['total mintage']
    increases = []
    for i in range(1, len(mintage_series)):
        prev = mintage_series.iloc[i-1]
        curr = mintage_series.iloc[i]
        increase = ((curr - prev) / prev) * 100
        increases.append(increase)
    
    avg_increase = sum(increases) / len(increases)
    print(f"Final Answer: {avg_increase:.1f}")