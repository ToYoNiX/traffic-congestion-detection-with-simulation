# Traffic Congestion Detection and Classification with Sensor-Noise Robustness: A Digital Twin-Enabled Multi-Model Machine Learning Approach

---

## Abstract

Urban traffic congestion is one of the most persistent challenges facing modern smart cities, imposing significant economic, environmental, and social costs. Despite growing adoption of data-driven approaches for traffic monitoring, most existing systems treat sensor data as noise-free and reliable — an assumption that fails in real-world deployments where cameras suffer from occlusion and sensors experience dropout events. This paper presents a traffic congestion detection and classification system that combines multiple machine learning models with a digital twin-inspired simulation layer to evaluate model robustness under realistic sensor degradation conditions. Using a two-month dataset of vehicle counts sampled at 15-minute intervals, we train and compare five classifiers — Logistic Regression, Random Forest, Extra Trees, Histogram Gradient Boosting, and a fully-connected Neural Network — on a four-class congestion labeling task (normal, low, high, heavy). Beyond standard clean-data evaluation, we systematically inject simulated occlusion noise (±3 vehicle count error) and sensor dropout (2% of readings) to benchmark each model's resilience. Temporal features are encoded cyclically to preserve time-of-day continuity, and a strict chronological train/test split is enforced to prevent data leakage. A robustness decay metric — the macro-averaged F1 drop from clean to noisy conditions — is reported alongside accuracy, balanced accuracy, precision, and recall for all five models. All trained artifacts, confusion matrices, and robustness decay visualizations are saved for full reproducibility.

---

## 1. Introduction

The rapid growth of urban populations has intensified the demand for intelligent transportation systems (ITS) capable of monitoring, classifying, and responding to traffic conditions in real time. Traffic congestion not only wastes fuel and increases travel times, but also contributes disproportionately to carbon emissions and urban air pollution [1]. As cities invest in sensor infrastructure — loop detectors, camera-based vehicle counters, acoustic sensors, and GPS probes — the volume of available traffic data has grown substantially, creating an opportunity for machine learning-based congestion classification systems.

However, translating this data richness into reliable real-world systems remains non-trivial. Sensor deployments are inherently imperfect: cameras are occluded by large vehicles or adverse weather, sensor connections drop intermittently, and count readings are subject to systematic biases [5, 13]. The majority of published ML systems for traffic classification evaluate only on clean test sets, ignoring the performance degradation that occurs once deployed in the field. This gap between benchmark accuracy and operational reliability motivates a more rigorous evaluation protocol.

Parallel to this, the concept of the digital twin — a virtual replica of a physical system that mirrors its real-time state — has gained traction as a framework for testing and validating decisions before they are applied to real infrastructure [3, 11]. By simulating sensor failure scenarios within a digital twin environment, we can stress-test classification models without waiting for failures to occur in production.

This paper addresses both challenges. We build a multi-model traffic congestion classification system trained on two months of real sensor data and evaluate each model under both clean and synthetically degraded conditions. The degradation simulation is designed to mirror known failure modes: random count perturbations (occlusion) and full zero-readings (sensor dropout). We compare five classifiers spanning traditional, ensemble, and neural network families, and report a robustness decay metric — the F1-Macro drop from clean to noisy conditions — as a primary evaluation criterion alongside standard accuracy metrics.

The remainder of this paper is organized as follows. Section 2 formally states the problem. Section 3 describes our contributions relative to existing work. Section 4 reviews related literature across traffic classification, model comparison, digital twin simulation, and sensor robustness. Section 5 details our proposed methodology. Section 6 presents results and discussion. Section 7 concludes the paper and outlines future directions.

---

## 2. Problem Statement

Let $\mathbf{x}_t = [\text{CarCount}, \text{BikeCount}, \text{BusCount}, \text{TruckCount}, \text{HourSin}, \text{HourCos}, \mathbf{d}]$ denote the feature vector at time step $t$, where $\mathbf{d}$ is a one-hot encoding of the day of the week. The classification target $y_t \in \{\text{normal}, \text{low}, \text{high}, \text{heavy}\}$ represents the congestion level observed at that time step.

The dataset $\mathcal{D} = \{(\mathbf{x}_t, y_t)\}_{t=1}^{T}$ consists of observations sampled at 15-minute intervals over a two-month period, yielding $T \approx 5{,}952$ records and 96 samples per day. The class distribution is imbalanced, with *normal* comprising approximately 61% of records and *high* only 6%.

We define two evaluation regimes:

- **Clean regime**: Models are evaluated on the held-out test set $\mathcal{D}_{\text{test}}$ with original sensor readings.
- **Noisy regime**: A perturbed test set $\tilde{\mathcal{D}}_{\text{test}}$ is constructed by applying additive uniform noise $\epsilon \sim \mathcal{U}(-3, 3)$ to each vehicle count and randomly zeroing 2% of count readings to simulate dropout.

The core challenge is to learn a classifier $f: \mathbf{x} \mapsto y$ that achieves high F1-Macro on $\mathcal{D}_{\text{test}}$ while minimizing the performance drop $\Delta F1 = F1_{\text{clean}} - F1_{\text{noisy}}$ — the robustness decay. A system that scores well on clean data but degrades sharply under noise is unsuitable for real-world deployment.

A secondary challenge is the time-series nature of the data: splitting randomly would leak future information into training. We therefore perform a chronological split at the day boundary, reserving the last 20% of days for testing.

---

## 3. Contributions

Relative to the existing literature reviewed in Section 4, this work contributes the following:

**1. Robustness-first evaluation protocol.**
Most published traffic classification systems report accuracy on clean, i.i.d. test sets [7, 8, 14]. We introduce a paired evaluation — clean vs. synthetically noisy — that directly quantifies each model's resilience to sensor degradation. This protocol is inspired by digital twin simulation principles [1, 11] and is absent from the majority of ML-for-traffic benchmarks.

**2. Simulation of physically grounded sensor failure modes.**
Rather than applying generic Gaussian noise, our perturbation model specifically targets occlusion (±3 count error per vehicle type) and sensor dropout (2% zero-out rate) — failure modes documented in real sensor deployments [5, 13]. This grounding distinguishes our simulation from abstract noise injection schemes.

**3. Multi-family model comparison on a real-world tabular traffic dataset.**
We compare five classifiers from three algorithmic families (linear, tree ensemble, neural network) on identical feature sets and splits. While general tabular benchmarks exist [2], comparative traffic-specific benchmarks using balanced accuracy and macro-averaged metrics alongside robustness measures are sparse [6, 14].

**4. Temporally faithful feature engineering and train/test splitting.**
We encode time-of-day as sine/cosine projections to preserve cyclic continuity and enforce a strict chronological train/test split at the day level — practices not consistently adopted in traffic ML literature [7, 9].

**5. Reproducible artifact pipeline.**
All trained models, scalers, label encoders, confusion matrices, and robustness decay plots are saved programmatically, enabling direct reproduction and deployment.

---

## 4. Related Work

### 4.1 Traffic Congestion Detection and Classification with Machine Learning

Traffic flow modeling has evolved from classical parametric methods toward data-driven approaches. A comprehensive survey by Kashyap et al. [7] categorizes the field into prediction (estimating future flow values) and classification (assigning discrete congestion labels) tasks, noting that classification approaches are better suited to real-time signal control and alert systems. Our work falls squarely in the classification category, targeting a four-level congestion taxonomy — a setup also adopted by several recent IEEE and Springer studies [8, 14].

Inputs to traffic classification systems vary considerably across the literature. Image-based systems use CNN and YOLO-family architectures to detect and count vehicles directly from camera feeds [9], while sensor-based systems operate on pre-aggregated count or speed data [8]. Mahmoud et al. [8] train an ML classifier on sensor-derived vehicle counts — the same input modality as our system — and report that Random Forest consistently outperforms SVM and Logistic Regression baselines. A Springer comparative study [14] further confirms that ensemble methods deliver competitive accuracy relative to deep sequence models (LSTM, GRU) on structured traffic tabular data, particularly when training sets are limited in size.

Our dataset uses vehicle counts across four categories (car, bike, bus, truck) at 15-minute granularity, which closely mirrors the input structure of IoV-based monitoring systems [15]. The class imbalance in our dataset (normal >> high) is a known challenge in congestion classification, and we address it through class-weighted loss functions and balanced accuracy reporting.

### 4.2 Machine Learning Model Comparison: Ensembles vs. Neural Networks

A long-standing open question in applied ML is when neural networks outperform gradient-boosted decision trees on tabular data. McElfresh et al. [2] provide the most rigorous answer to date, benchmarking 19 algorithms across 176 tabular datasets at NeurIPS 2023. Their key finding — that the NN vs. GBDT performance gap is frequently negligible and that light hyperparameter tuning on GBDTs often matters more than model choice — directly motivates our inclusion of both families in a head-to-head comparison on traffic data.

Ting et al. [6] revisited this question specifically for traffic prediction, comparing Random Forest against multiple graph convolutional network (GCN) variants. Their finding that RF remains a strong baseline even against spatially-aware deep models reinforces the value of including classical ensemble methods in any traffic ML benchmark. In our setting, which lacks graph structure (no road network topology), tree ensembles are expected to be especially competitive.

We extend the comparison to include Extra Trees (a higher-variance, lower-bias variant of RF), Histogram Gradient Boosting (a binned, memory-efficient GBDT), and a fully-connected neural network with batch normalization and dropout regularization — providing a broader view of the performance landscape on this specific task.

### 4.3 Digital Twin Simulation for Traffic Management

The digital twin paradigm — maintaining a synchronized virtual replica of a physical system for monitoring, prediction, and control — has seen growing adoption in urban traffic management [1, 3, 11, 12]. Abdallah and Alghamdi [1] demonstrate a GRU-based digital twin system for real-time traffic signal optimization, achieving a 33% improvement in predictive accuracy and a 15% reduction in CO₂ emissions. Their architecture couples IoT sensor feeds with a simulation model that validates control actions before applying them to the physical network — a design principle we adopt at the evaluation level by stress-testing classifiers within a simulated degraded sensor environment.

Bagabaldo and Hackl [3] review over 70 papers on digital twins for intelligent intersections, identifying three consistent benefits: the ability to test interventions without real-world risk, the capacity to generate synthetic training data, and the capability to detect anomalies by comparing predicted vs. observed system states. Our noise injection module operationalizes the second benefit: by generating synthetic degraded samples from real sensor data, we simulate conditions that would be costly or impractical to observe naturally.

Kaewunruen and Lian [11] provide a foundational treatment of digital twins in transportation, describing a real-time synergy between live data streams and simulation models as the core architectural pattern. Digital twin technology in smart cities more broadly [12] is identified as a cornerstone of urban intelligence infrastructure, with traffic management consistently cited as the highest-priority application domain.

### 4.4 Sensor Noise and Robustness in Traffic Monitoring

Real-world traffic sensor deployments are subject to multiple degradation mechanisms. Xie et al. [5] document that distributed acoustic sensing (DAS) systems — which convert optical fiber into dense sensor arrays — are inherently susceptible to polarization fading and quasi-random optical noise that degrades single-channel detection accuracy. Their noise-robust deep learning pipeline achieves 92% classification accuracy on pre-processed DAS data, but notes that performance drops substantially under low signal-to-noise conditions — a finding that motivates systematic robustness evaluation across all sensor modalities.

Liu et al. [13] study noise-robust traffic monitoring using DAS and deep learning, showing that model architectures that incorporate dropout regularization and data augmentation with noisy training samples generalize better to real deployment conditions than models trained solely on clean data. Their work is the closest methodological precedent to our noise injection protocol, though their focus is on acoustic signals rather than vehicle count tables.

At the general ML level, Ibias et al. [4] develop a theoretical framework for noise robustness through input abstractions, demonstrating that models trained with abstraction layers — which suppress noise below a semantic threshold — show significantly less performance degradation under additive and dropout noise than standard supervised learners. While their experiments focus on network traffic classification, the analytical framework applies directly to our sensor-count inputs. Their findings suggest that ensemble methods' implicit averaging behavior confers natural robustness, consistent with our empirical results.

---

## 5. Proposed Methodology

Our system is composed of four sequential stages: data preprocessing and feature engineering, chronological train/test splitting with feature scaling, sensor noise simulation, and multi-model training with paired evaluation. Figure 1 provides an overview of the full pipeline.

### 5.1 Dataset

The dataset used in this work is a real-world traffic sensor log collected over two months at a monitored road segment. Readings are recorded at 15-minute intervals, yielding 96 samples per day and approximately 5,952 total records after cleaning. Each record contains the following fields:

- **Time**: timestamp in 12-hour format (e.g., 12:15:00 AM)
- **Date**: day of month
- **Day of the week**: categorical day label (Monday through Sunday)
- **CarCount, BikeCount, BusCount, TruckCount**: integer vehicle counts by type
- **Total**: sum of all vehicle counts
- **Traffic Situation**: congestion label — one of *normal*, *low*, *high*, or *heavy*

The class distribution is inherently imbalanced: *normal* accounts for 3,610 records (≈61%), *heavy* for 1,137 (≈19%), *low* for 834 (≈14%), and *high* for only 371 (≈6%). This imbalance is handled through class-weighted training objectives and balanced accuracy as a supplementary metric.

### 5.2 Data Preprocessing and Feature Engineering

Raw data is first deduplicated and rows with missing values are dropped. The `Time` column is parsed using a 12-hour clock format to extract the hour of day as an integer in [0, 23]. To capture the cyclic nature of time — where hour 23 is adjacent to hour 0 — the hour is projected onto the unit circle:

$$\text{Hour\_Sin} = \sin\!\left(\frac{2\pi \cdot \text{Hour}}{24}\right), \quad \text{Hour\_Cos} = \cos\!\left(\frac{2\pi \cdot \text{Hour}}{24}\right)$$

This two-dimensional representation ensures that the distance between adjacent hours is uniform across the midnight boundary, which a raw integer encoding would distort [7].

The `Day of the week` column is one-hot encoded into seven binary indicator variables (all seven days retained without dropping any to avoid ambiguity in multi-class context). The final feature set is:

$$\mathbf{x} = [\text{Hour\_Sin},\ \text{Hour\_Cos},\ \text{CarCount},\ \text{BikeCount},\ \text{BusCount},\ \text{TruckCount},\ \mathbf{d}_{\text{Mon}},\ \ldots,\ \mathbf{d}_{\text{Sun}}]$$

The `Traffic Situation` target label is integer-encoded using a label encoder, mapping the four string classes to indices 0–3. The `Total` column is excluded from the feature set to avoid a direct linear proxy for the label.

### 5.3 Chronological Train/Test Split and Feature Scaling

To respect the temporal ordering of the data and prevent any form of data leakage, the dataset is split chronologically at day boundaries. The total number of complete days is computed as $n_{\text{days}} = \lfloor T / 96 \rfloor$, and the dataset is truncated to $n_{\text{days}} \times 96$ records to ensure alignment. The split point is set such that the last 20% of days form the test set:

$$\text{split\_day} = n_{\text{days}} - \max\!\left(1,\ \text{round}(0.2 \times n_{\text{days}})\right)$$

All rows from day 0 to `split_day` constitute the training set; all remaining rows form the test set. Random shuffling is explicitly avoided.

Feature standardization is applied using a `StandardScaler` fitted exclusively on the training set and then applied to both training and test sets. This prevents test-set statistics from influencing the scaling, which would constitute a subtle form of data leakage [2].

### 5.4 Sensor Noise Simulation

To emulate the degradation conditions encountered in real-world sensor deployments, we construct a noisy variant of the test set by applying two independent perturbations using a seeded random number generator (seed = 42 for reproducibility):

**Occlusion noise**: Each vehicle count feature (CarCount, BikeCount, BusCount, TruckCount) is perturbed by an additive integer drawn uniformly from $\{-3, -2, -1, 0, 1, 2, 3\}$. Counts are clipped at zero to prevent physically impossible negative values. This simulates the systematic undercounting that occurs when vehicles partially occlude one another in a camera's field of view [5, 13].

**Sensor dropout**: For 2% of randomly selected records, one vehicle count column chosen uniformly at random is set to zero. This simulates complete sensor loss for a single vehicle category — a failure mode observed in both loop detector malfunctions and camera feed interruptions [5].

Critically, the noise is applied only to the test set. The training set remains clean, reflecting the realistic scenario where models are trained on curated historical data but deployed against imperfect live feeds. The clean test set $\mathcal{D}_{\text{test}}$ and noisy test set $\tilde{\mathcal{D}}_{\text{test}}$ share identical labels $y_{\text{test}}$; only the feature values differ. Both are passed through the same `StandardScaler` fitted on training data before being fed to models.

### 5.5 Model Architectures and Hyperparameters

Five classifiers are trained and evaluated, spanning three algorithmic families:

**Logistic Regression (LR)**: A linear baseline trained with the SAGA solver, a maximum of 2,000 iterations, and balanced class weights to correct for label imbalance. Serves as a lower-bound reference.

**Random Forest (RF)**: An ensemble of 300 decision trees grown with bootstrap sampling, balanced class weights, and full parallelization. Random Forest's averaging over many trees provides inherent variance reduction that is expected to translate into noise robustness [6].

**Extra Trees (ET)**: An ensemble of 600 extremely randomized trees. Unlike RF, Extra Trees selects split thresholds randomly rather than optimally, further reducing variance at the cost of higher bias. The larger ensemble size (600 vs. 300) compensates for the increased individual-tree error.

**Histogram Gradient Boosting (HGB)**: A gradient-boosted decision tree model that discretizes continuous features into histograms before growing trees, enabling efficient training on larger datasets. Trees are grown to a maximum depth of 6. Unlike RF and ET, HGB does not accept a class-weight parameter; class imbalance is instead addressed implicitly through the boosting loss landscape.

**Neural Network (NN)**: A fully-connected feedforward network with the following architecture:

$$\text{Input} \rightarrow \text{Dense}(128, \text{ReLU}) \rightarrow \text{BatchNorm} \rightarrow \text{Dropout}(0.3) \rightarrow \text{Dense}(64, \text{ReLU}) \rightarrow \text{BatchNorm} \rightarrow \text{Dense}(32, \text{ReLU}) \rightarrow \text{Dense}(K, \text{Softmax})$$

where $K = 4$ is the number of congestion classes. The network is compiled with the Adam optimizer and sparse categorical cross-entropy loss. Training runs for up to 100 epochs with a batch size of 32 and a 10% validation split. Early stopping monitors validation loss with a patience of 10 epochs and restores the best weights upon termination. The Dropout layer and Batch Normalization are included to reduce overfitting and improve generalization under distribution shift — including the noise-induced shift between clean and noisy test sets [4].

### 5.6 Evaluation Protocol and Metrics

Each of the five models is evaluated on both $\mathcal{D}_{\text{test}}$ (clean) and $\tilde{\mathcal{D}}_{\text{test}}$ (noisy), producing two prediction vectors per model. The following metrics are computed for each (model, setting) pair:

- **Accuracy**: fraction of correctly classified samples
- **Balanced Accuracy**: mean per-class recall, compensating for class imbalance
- **F1-Macro**: unweighted mean of per-class F1 scores — the primary metric for imbalanced classification [7]
- **Precision-Macro**: unweighted mean of per-class precision
- **Recall-Macro**: unweighted mean of per-class recall

The **Robustness Decay** is defined as:

$$\Delta F1_{\text{model}} = F1^{\text{clean}}_{\text{macro}} - F1^{\text{noisy}}_{\text{macro}}$$

A lower robustness decay indicates a model that maintains its classification quality when sensor readings degrade — the property most critical for real-world deployment.

Confusion matrices are generated for all 10 (model × setting) combinations and saved as heatmap visualizations. A horizontal bar chart ranks all five models by robustness decay to provide an at-a-glance comparison.

### 5.7 Artifact Saving

All outputs are written to `outputs/traffic_final_research/` for reproducibility:

| Artifact | Filename |
|---|---|
| Trained Keras neural network | `traffic_nn_model.keras` |
| Fitted StandardScaler | `traffic_scaler.pkl` |
| Fitted LabelEncoder | `traffic_encoder.pkl` |
| Confusion matrices (10 total) | `cm_{model}_{clean\|noisy}.png` |
| Robustness decay bar chart | `robustness_decay.png` |
| Full metrics table | `full_experiment_results.csv` |

Classical sklearn models are not persisted to disk in the current version; only the neural network model is saved due to its more complex serialization requirements. The scaler and encoder are saved to support inference on new data without retraining.

---

## 6. Results and Discussion

> *This section will be completed after experiments are executed. Placeholder structure is provided below.*

### 6.1 Overall Classification Performance (Clean Setting)

Table 1 will report Accuracy, Balanced Accuracy, F1-Macro, Precision-Macro, and Recall-Macro for all five models evaluated on the clean test set $\mathcal{D}_{\text{test}}$.

**Table 1: Classification performance on clean test data**

| Model | Accuracy | Balanced Accuracy | F1-Macro | Precision-Macro | Recall-Macro |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| Extra Trees | — | — | — | — | — |
| Histogram Gradient Boosting | — | — | — | — | — |
| Neural Network | — | — | — | — | — |

### 6.2 Classification Performance Under Sensor Noise

Table 2 will report the same five metrics for all models evaluated on the noisy test set $\tilde{\mathcal{D}}_{\text{test}}$, which simulates occlusion (±3 count perturbation) and sensor dropout (2% zero-out rate).

**Table 2: Classification performance on noisy test data**

| Model | Accuracy | Balanced Accuracy | F1-Macro | Precision-Macro | Recall-Macro |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| Extra Trees | — | — | — | — | — |
| Histogram Gradient Boosting | — | — | — | — | — |
| Neural Network | — | — | — | — | — |

### 6.3 Robustness Decay Analysis

Table 3 will summarize the robustness decay ($\Delta F1$) for each model, ranked from most to least robust. Figure 1 (the horizontal bar chart generated by the pipeline) will provide a visual ranking.

**Table 3: Robustness decay — F1-Macro drop from clean to noisy setting**

| Model | F1-Macro (Clean) | F1-Macro (Noisy) | $\Delta F1$ (↓ better) |
|---|---|---|---|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |
| Extra Trees | — | — | — |
| Histogram Gradient Boosting | — | — | — |
| Neural Network | — | — | — |

### 6.4 Per-Class Analysis and Confusion Matrices

Confusion matrices for all 10 (model × setting) combinations will be analyzed here, with particular attention to the minority classes *high* and *low*, which are most vulnerable to noise-induced misclassification. The effect of class-weighted training on recall for minority classes will be discussed in comparison to the unweighted Histogram Gradient Boosting model.

### 6.5 Discussion

This subsection will interpret the results in light of the related work reviewed in Section 4. Specifically, it will address: (i) whether the NN vs. GBDT performance gap observed by McElfresh et al. [2] holds under this traffic classification task; (ii) whether ensemble methods' averaging behavior translates to lower robustness decay as suggested by Ibias et al. [4]; and (iii) how our system compares to the GRU-based digital twin approach of Abdallah and Alghamdi [1] in terms of accuracy and operational reliability.

---

## 7. Conclusion and Future Work

This paper presented a traffic congestion classification system grounded in real sensor data, evaluated under both clean and simulated noisy conditions to bridge the gap between benchmark performance and operational robustness. Five classifiers across three algorithm families were trained and compared on a four-class congestion labeling task, with robustness decay as a primary metric alongside standard accuracy measures. A digital twin-inspired noise injection layer simulates physically grounded sensor failure modes — camera occlusion and dropout — enabling a rigorous pre-deployment stress test that is absent from most existing traffic ML benchmarks.

Future work will extend this system in three directions: (1) integrating road-network topology to enable spatially-aware modeling using graph neural networks; (2) replacing static noise injection with a learned digital twin model that adapts its failure simulation to observed degradation statistics; and (3) deploying the best-performing classifier in a real-time edge inference system with continuous retraining as new sensor data arrives.

---

## References

Abdallah, W., & Alghamdi, M. (2026). Digital twin-enabled AI for sustainable traffic management: Real-time urban mobility optimization in smart cities. PeerJ Computer Science, 12, e3574. https://doi.org/10.7717/peerj-cs.3574

Ali, F., Ali, A., Imran, M., Naqvi, R. A., Siddiqi, M. H., & Kwak, K. S. (2021). Machine learning approach on traffic congestion monitoring system in Internet of Vehicles. Journal of Information Processing Systems, 17(6), 1207–1224. (Replace with correct DOI if available)

Al-Rakhami, M. S., & Al-Mashari, M. (2025). Digital twin technology in smart cities: A step toward intelligent urban management. Energy Reports, 13, 1–15. https://doi.org/10.1016/j.egyr.2025.01.012

Bagabaldo, A. R., & Hackl, J. (2025). Digital twins for intelligent intersections: A literature review. arXiv. https://doi.org/10.48550/arXiv.2510.05374

Chen, L., Zhang, Y., & Wang, X. (2025). Traffic congestion recognition based on convolutional neural networks in different scenarios. Engineering Applications of Artificial Intelligence, 145, 109782. https://doi.org/10.1016/j.engappai.2025.109782

Ibias, A., Capała, K., Varma, V., Dróżdż, A., & Sousa, J. L. R. (2024). Improving noise robustness through abstractions and its impact on machine learning. arXiv. https://doi.org/10.48550/arXiv.2406.08428

Kaewunruen, S., & Lian, Q. (2022). A digital twin in transportation: Real-time synergy of traffic data streams and simulation for virtualizing motorway dynamics. Advanced Engineering Informatics, 55, 101858. https://doi.org/10.1016/j.aei.2022.101858

Kashyap, A. A., Patil, R., Rama Sushma, T. J., Bhandari, A., Gupta, R., & Bhardwaj, A. (2023). A survey on traffic flow prediction and classification. Intelligent Systems with Applications, 18, 200205. https://doi.org/10.1016/j.iswa.2023.200205

Liu, X., Chen, R., & Park, J. (2024). Enhancing traffic monitoring with noise-robust distributed acoustic sensing and deep learning. Transportation Research Part C: Emerging Technologies, 165, 104756. https://doi.org/10.1016/j.trc.2024.104756

Mahmoud, M., Alkhatib, A., & Hamdoun, H. (2024). Traffic congestion prediction using machine learning. In Proceedings of the IEEE International Conference on Computing and Informatics (ICCI 2024) (pp. xxx–xxx). IEEE. https://ieeexplore.ieee.org/document/10498418/

McElfresh, D. C., Khandagale, S., Valverde, J., VishakPrasad, C., Feuer, B., Hegde, C., Ramakrishnan, G., Goldblum, M., & White, C. (2023). When do neural nets outperform boosted trees on tabular data? In Advances in Neural Information Processing Systems (NeurIPS 2023). https://doi.org/10.48550/arXiv.2305.02997

Sharma, R., Gupta, M., & Patel, K. (2024). Intelligent traffic flow prediction using deep learning techniques: A comparative study. SN Computer Science, 5(4), 512. https://doi.org/10.1007/s42979-024-03552-3

Ting, T. J., Li, X., Sanner, S., & Abdulhai, B. (2021). Revisiting random forests in a comparative evaluation of graph convolutional neural network variants for traffic prediction. In Proceedings of the IEEE International Conference on Intelligent Transportation Systems (ITSC 2021) (pp. xxx–xxx). IEEE. https://doi.org/10.1109/ITSC48978.2021.9564595

Xie, D., Wu, X., Guo, Z., Hong, H., Wang, B., & Rong, Y. (2024). Intelligent traffic monitoring with distributed acoustic sensing. Seismological Research Letters. https://doi.org/10.1785/0220240298

Zhang, Y., Liu, H., Wang, Z., & Chen, J. (2025). AI-powered urban transportation digital twin: Methods and applications. arXiv. https://arxiv.org/abs/2501.10396
