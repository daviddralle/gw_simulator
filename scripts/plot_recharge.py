import pandas as pd
import matplotlib.pyplot as plt
import sys

df = pd.read_csv(sys.argv[1])
df['date'] = pd.to_datetime(df['date'])

fig, ax = plt.subplots(figsize=(15, 5))
ax.bar(df['date'], df['Recharge'], color='blue', alpha=0.5, width=1.5)
ax.set_ylabel('Recharge (mm/day)')
ax.set_title('Daily Recharge Forcing (2001 - 2024)')
ax.invert_yaxis()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig('outputs/recharge_forcing_2001_2024.png', dpi=300)
print("Recharge plot saved to outputs/recharge_forcing_2001_2024.png")
