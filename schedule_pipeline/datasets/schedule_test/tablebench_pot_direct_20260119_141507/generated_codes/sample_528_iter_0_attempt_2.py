# Extract values for Constitutional Democratic Party (Kadets)
first_duma_seats = 179
second_duma_seats = 92
total_first_duma = 566
total_second_duma = 453

# Calculate seat shares as percentages
first_share = (first_duma_seats / total_first_duma) * 100
second_share = (second_duma_seats / total_second_duma) * 100

# Compute change
seat_share_change = second_share - first_share

print(f"Final Answer: {seat_share_change:.2f}")