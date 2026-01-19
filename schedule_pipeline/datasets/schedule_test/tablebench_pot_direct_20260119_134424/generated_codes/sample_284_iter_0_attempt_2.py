import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where issue price is $34.95
filtered_df = df[df['issue price'] == '34.95']
# Convert 'total mintage' to integer
filtered_df['total mintage'] = filtered_df['total mintage'].astype(int)
# Calculate year-over-year percentage increase
percentage_increases = []
for i in range(1, len(filtered_df)):
    prev_mintage = filtered_df['total mintage'].iloc[i-1]
    curr_mintage = filtered_df['total mintage'].iloc[i]
    increase = ((curr_mintage - prev_mintage) / prev_mintage) * 100
    percentage_increases.append(increase)
# Calculate average percentage increase
avg_increase = sum(percentage_increases) / len(percentage_increases)
print(f"Final Answer: {avg_increase:.2f}")