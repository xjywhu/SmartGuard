## Threshold (A2)

![Method](../figures/threshold.png)

We tested the anomaly detection performance of SmartGuard under different $th$. As shown in the above figure, a smaller threshold may cause the model to misidentify normal samples as anomalies, while a larger threshold (>95\%) may cause the model to miss anomalies. When the quantile of threshold is 95%, SmartGuard can achieve the best anomaly detection performance.