import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract values for Constitutional Democratic Party (Kadets)
first_duma_seats = 179
second_duma_seats = 92
first_duma_total = 566
second_duma_total = 453

# Calculate percentages
first_percent = (first_duma_seats / first_duma_total) * 100
second_percent = (second_duma_seats / second_duma_total) * 100

# Calculate the change (difference)
seat_share_change = second_percent - first_percent

print(f"Final Answer: {seat_share_change:.2f}")