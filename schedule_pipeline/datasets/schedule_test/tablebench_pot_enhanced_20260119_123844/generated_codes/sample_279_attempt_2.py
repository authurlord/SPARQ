# Calculate total estimated exposure
activity_10g = 1834
exposure_10g = 0.03

activity_01g = 18
exposure_01g = 0.0  # from table, but we ignore it and use proportionality

# Since exposure is proportional to activity, use the 10g data to find rate
rate = exposure_10g / activity_10g

# Total activity
total_activity = activity_10g + activity_01g

# Total exposure
total_exposure = total_activity * rate

print(f"Final Answer: {total_exposure:.3f}")