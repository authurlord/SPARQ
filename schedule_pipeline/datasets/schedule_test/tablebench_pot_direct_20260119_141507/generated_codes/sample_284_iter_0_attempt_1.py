import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Filter rows where issue price is $34.95
filtered_df = df[df['issue price'] == '34.95'].copy()

# Sort by year to ensure chronological order
filtered_df = filtered_df.sort_values('year').reset_index(drop=True)

# Extract total mintage values
mintage_values = filtered_df['total mintage'].astype(int)

# Compute percentage increase from one year to the next
# If there's only one row, no percentage increase can be calculated
if len(mintage_values) < 2:
    print("Final Answer: 0")
else:
    # Calculate percentage increases
    percentage_increases = []
    for i in range(1, len(mintage_values)):
        prev_mintage = mintage_values[i - 1]
        curr_mintage = mintage_values[i]
        increase = ((curr_mintage - prev_mintage) / prev_mintage) * 100
        percentage_increases.append(increase)
    
    # Average the increases
    avg_increase = sum(percentage_increases) / len(percentage_increases)
    print(f"Final Answer: {avg_increase:.1f}")