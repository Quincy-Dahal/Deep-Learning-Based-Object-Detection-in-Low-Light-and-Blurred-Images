import pandas as pd

df = pd.read_csv("Results/final_model_comparison.csv")

summary = df.groupby("model", as_index=False).agg({
    "AP_50_95": "mean",
    "AP_50": "mean",
    "F1_score": "mean",
    "Recall": "mean",
    "Precision": "mean"
})

summary = summary.rename(columns={
    "model": "Model",
    "AP_50_95": "Avg_AP_50_95",
    "AP_50": "Avg_AP_50",
    "F1_score": "Avg_F1",
    "Recall": "Avg_Recall",
    "Precision": "Avg_Precision"
})

summary = summary.round(3)

summary.to_csv("Results/model_average_summary.csv", index=False)

print("Saved: Results/model_average_summary.csv")
print(summary)