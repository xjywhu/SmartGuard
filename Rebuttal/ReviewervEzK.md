## Literature Survey (A2)

\subsection{User Behavior Modeling in Smart Homes}
Some works propose to model user behavior (i.e., user device interaction) based on deep learning.
~\cite{iotgaze} uses event transition graph to model IoT context and detect anomalies. In~\cite{dsnGraph}, authors build device interaction graph to learn the device state transition relationship caused by user actions. ~\cite{hawatcher} detects anomalies through correlational analysis of device actions and physical environment. ~\cite{DBLP:conf/huc/SrinivasanSW08} infers user behavior through readings from various sensors installed in the user's home.
IoTBeholder \cite{zou2023iotbeholder} utilizes attention-based LSTM to predict the user behavior from history sequences. SmartSense \cite{jeon2022accurate} leverages query-based transformer to model contextual information of user behavior sequences. DeepUDI \cite{xiao2023user} and SmartUDI \cite{xiao2023know} use relational gated graph neural networks, capsule neural networks and contrastive learning to model users' routines, intents and multi-level periodicities. However, above methods aim at predicting next behavior of user accurately, they can not be applied into abnormal behavior detection.


\subsection{Attacks and Defenses in Smart Homes}
% Due to vulnerabilities arising from both IoT devices and the architecture of IoT systems, 
An increasing number of attack vectors have been identified in smart homes in recent years. In addition to cyber attacks, it is also a concerning factor that IoT devices are often close association with the user's physical environment and they have the ability to alter physical environment. In this context, the automation introduces more serious security risks. Prior research has revealed that adversaries can leak personal information, and gain physical access to the home~\cite{ContexloT, DBLP:conf/uss/CelikBSATMU18}. In~\cite{spoof}, spoof attack is employed to exploit automation rules and trigger unexpected device actions. ~\cite{delay-sp, delay-dsn} apply delay-based attacks to disrupt cross-platform IoT information exchanges, resulting in unexpected interactions, rendering IoT devices and smart homes in an insecure state. This series of attacks aim at causing smart home devices to exhibit expected actions, thereby posing significant security threats. Therefore, designing an effective mechanism to detect such attacks is necessary. 6thSense \cite{sikder20176thsense} utilizes Naive Bayes to detect malicious behavior associated with sensors in smart homes. Aegis \cite{Siker19Aegis} utilizes a Markov Chain to detect malicious behaviors. ARGUS \cite{Rieger23ARGUS} designed an Autoencoder based on Gated Recurrent Units (GRU) to detect infiltration attacks. However, these methods ignore the behavior imbalance, temporal information and noise behaviors.


We also add some baselines:
[R1] Madan N, Ristea N C, Ionescu R T, et al. Self-supervised
masked convolutional transformer block for anomaly detection[J].
IEEE Transactions on Pattern Analysis and Machine Intelligence,
2023.
[R2] Xu J, Wu H, Wang J, et al. Anomaly Transformer: Time Se-
ries Anomaly Detection with Association Discrepancy[C]//International
Conference on Learning Representations. 2021

[R3] Dai, Xuan, et al. "Homeguardian: Detecting anomaly events in smart home systems." Wireless Communications and Mobile Computing 2022 (2022).

[R4] Tang, Sihai, et al. "Smart home IoT anomaly detection based on ensemble model learning from heterogeneous data." 2019 IEEE International Conference on Big Data (Big Data). IEEE, 2019.


In our paper, we have many

## Experiments (A3)

(1) Ablation study. As shown in the following tables, each
component of SmartGuard has a positive impact on results. The
combination of all components brings the best results on FR and SP dataset.

![Method](../figures/ablation_FR.png)
![Method](../figures/ablation_SP.png)


(2) Parameter experiments. As shown in Figure(a), when mask ratio is 0.8, step w/o mask is 6, SmartGuard achieves the best anomaly detection performance on FR dataset. As shown in Figure(b), on SP dataset, when mask ratio is 0.6, step w/o mask is 6, SmartGuard achieves the best performance.

![Method](../figures/mask_para.png)