# Extract the values of 'l (mh / km)' at frequency 1 Hz and 100k Hz
l_1_hz = 0.6129
l_100k_hz = 0.5807

# Calculate percentage change
percentage_change = ((l_100k_hz - l_1_hz) / l_1_hz) * 100

print(f"Final Answer: {percentage_change:.2f}%")