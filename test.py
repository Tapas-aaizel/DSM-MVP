# import pandas as pd
# import matplotlib.pyplot as plt

# # Load both CSV files
# df1 = pd.read_csv("/home/tapas/Solar-MVP-DSM-v1/results/DSM/15-02-2026/hindcast_audit.csv")
# df2 = pd.read_csv("/home/tapas/Solar-MVP-DSM-v1/results/DSM/16-02-2026/hindcast_audit.csv")

# # Check columns (optional)
# print(df1.columns)
# print(df2.columns)

# # Compute differences
# forecast_diff = df1["forecast_mw"] - df2["forecast_mw"]
# actual_diff   = df1["actual_mw"] - df2["actual_mw"]

# # Create a result dataframe (optional but nice)
# diff_df = pd.DataFrame({
#     "forecast_diff": forecast_diff,
#     "actual_diff": actual_diff
# })

# print(diff_df.head())

# # 📈 Plot differences
# plt.figure(figsize=(10,5))
# plt.plot(diff_df["forecast_diff"], label="Forecast MW Difference")
# plt.plot(diff_df["actual_diff"], label="Actual MW Difference")

# plt.axhline(0)   # zero reference line
# plt.xlabel("Time Index")
# plt.ylabel("MW Difference")
# plt.title("Difference Between Two CSV Files")
# plt.legend()
# plt.grid(True)
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs
df1 = pd.read_csv("/home/tapas/Solar-MVP-DSM-v1/results/DSM/14-02-2026/hindcast_audit.csv")
df2 = pd.read_csv("/home/tapas/Solar-MVP-DSM-v1/results/DSM/16-02-2026/hindcast_audit.csv")


# Create diff dataframe
df_diff = pd.DataFrame()

df_diff["forecast_diff"] = df1["forecast_mw"] - df2["forecast_mw"]
df_diff["actual_diff"]   = df1["actual_mw"] - df2["actual_mw"]

# Error percentage (df2 as reference)
df_diff["forecast_error_pct"] = (df_diff["forecast_diff"] / df2["forecast_mw"]) * 100
df_diff["actual_error_pct"]   = (df_diff["actual_diff"] / df2["actual_mw"]) * 100

# Round to 3 decimals
df_diff = df_diff.round(3)

# 💾 Save to CSV
df_diff.to_csv("mw_difference_and_error.csv", index=False)

print("Saved successfully → mw_difference_and_error.csv")

# Optional plot
plt.figure(figsize=(10,5))
plt.plot(df_diff["forecast_diff"], label="Forecast Diff")
plt.plot(df_diff["actual_diff"], label="Actual Diff")
plt.axhline(0)
plt.legend()
plt.show()
