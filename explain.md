# Notebook Explanation: models_train.ipynb

Everything in the notebook, section by section — what it does, why it was done that way, and why that specific choice was made over alternatives.

---

## 1. Imports & Environment Setup

### `SEED = 42`
Every random component in the experiment (train/test split, tree-based model splits, neural network weight initialization, noise injection) receives this seed. Without it, results change on every run, making it impossible to reproduce or compare runs.

### `ROWS_PER_DAY = 96`
The dataset records vehicle counts every 15 minutes. `24 hours × 4 readings/hour = 96 rows per day`. This constant drives the time-based split logic — it's how we define what a "day" is in the data.

### `os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"`
Silences TensorFlow's C++ backend warnings (GPU/CPU optimization notices, kernel loading messages). They're irrelevant to the experiment and would flood the notebook output.

### `%matplotlib inline`
Jupyter magic that renders plots directly in the cell output. The original script used `matplotlib.use("Agg")` which is the headless backend for saving files without a display — correct for scripts, wrong for notebooks where you actually want to see the plots.

---

## 2. Data Preprocessing & Feature Engineering

### `drop_duplicates().dropna()`
Basic cleaning first, before any feature creation. Sensor systems often emit duplicate readings (network retransmits, logging bugs) and occasionally drop values entirely. Deduplication happens before NaN removal to avoid incorrectly dropping rows that look like duplicates only because of missing values.

### Hour extraction with `%I:%M:%S %p`
The "Time" column is in 12-hour format with AM/PM (e.g., `08:30:00 AM`). The format string `%I:%M:%S %p` matches exactly that. Using `%H` would fail on this data.

### `Hour_Sin` and `Hour_Cos` instead of raw `Hour`
This is cyclical encoding. Hour is a circular feature — hour 23 and hour 0 are only 1 hour apart in reality, but as raw integers they look 23 units apart. Any model treating integers as linear distances would think midnight is the furthest possible time from 11pm.

The sin/cos transformation maps the 24-hour cycle onto a unit circle:

```
Hour_Sin = sin(2π × Hour / 24)
Hour_Cos = cos(2π × Hour / 24)
```

Together, these two values uniquely and continuously encode any hour of the day with correct neighbors. Hour 0 and hour 23 end up geometrically close. The raw `Hour` column is not kept as a feature because it would reintroduce the discontinuity.

### `pd.get_dummies` on "Day of the week" with `drop_first=False`
One-hot encodes the day (Monday, Tuesday, ...) into 7 binary columns. `drop_first=False` keeps all 7 rather than dropping one to avoid the dummy variable trap. For tree-based models this doesn't matter (they aren't affected by multicollinearity). For logistic regression, regularization handles the collinearity. Keeping all 7 makes each day's effect explicit and makes the feature matrix easier to inspect.

### `get_features`: what's included and what's not
The selected features are: `Hour_Sin`, `Hour_Cos`, `CarCount`, `BikeCount`, `BusCount`, `TruckCount`, plus all the day-of-week dummies.

What's explicitly excluded:
- The original `Time` string — it's been replaced by `Hour_Sin`/`Hour_Cos`
- The raw `Hour` integer — replaced by its cyclic encoding
- `Traffic Situation` — the target label, not a feature
- Any ID or date columns — not predictive of the current reading

---

## 3. Time-Based Split & Noise Injection

### Why a time-based split instead of `train_test_split`
Traffic data is a time series. A random split would let the model train on data from next week and predict last week's data, which is data leakage. Real deployment only ever has past data to predict the future. A chronological split — all training data comes before all test data — is the only valid approach here.

### Why split on whole days
The split is calculated in units of `ROWS_PER_DAY` so that no day is split across train and test. A partial day at the boundary could give the model context about the test period (e.g., knowing the morning of a day helps predict the evening of the same day). Keeping days whole prevents that.

### `max(1, int(round(n_days * test_size)))`
Guards against edge cases with very small datasets where `round(n_days * 0.2)` could yield 0, which would mean no test set at all. The `max(1, ...)` ensures at least one day ends up in test.

### `inject_noise`: what it simulates
Real roadside sensors fail in two main ways:

**Occlusion errors** (`rng.integers(-3, 4, size=len(Xn))`) — a vehicle partially blocks the sensor's view, causing it to miscount by ±1 to ±3. This is uniform integer noise centered near zero. `.clip(lower=0)` is applied because a vehicle count can never be negative.

**Sensor dropout** (`rng.random(len(Xn)) < 0.02`) — 2% of readings have one randomly chosen vehicle type completely zeroed out, simulating a sensor that goes offline, resets, or loses power for one interval. A single vehicle type per row is zeroed (not the entire row) because sensors are per-lane or per-category.

### Why noise is applied only to `X_test`, not `X_train`
The experiment question is: *given a model trained on clean historical data, how well does it hold up when deployed sensors start failing?* That's the realistic scenario — data in the database is clean, but live inference gets noisy inputs. Training on noise would be a different (and stronger) setting.

### `scaler.transform(inject_noise(X_test))`
Noise is injected into the raw counts first, then the scaler is applied. This matches the deployment order: raw sensor reading → noise corruption → normalization → model inference. Doing it the other way around (scale first, add noise) would produce noise in the wrong units.

---

## 4. Visualisation Utilities

### Confusion matrix with `magma` colormap
A confusion matrix shows the exact breakdown of where the model gets confused — which classes get predicted as which other classes. This is much more informative than a single accuracy number, especially for a 4-class problem where "High" misclassified as "Heavy" is very different from "Low" misclassified as "Heavy".

`magma` is a perceptually uniform colormap (no artificial luminance jumps), which makes it easy to spot high-error cells visually without misleading color transitions.

`plt.show()` is called after `plt.savefig()` so the plot both saves to disk and renders in the notebook cell output.

---

## 5. Neural Network Architecture

### Layer progression: 128 → 64 → 32 → output
This tapering structure is standard for tabular classification. Wide first layers learn many simple feature interactions; narrower subsequent layers compress those into increasingly abstract representations. Going wider in later layers provides no benefit on tabular data and increases overfitting risk.

### `BatchNormalization`
Normalizes the output of each dense layer within the current batch before passing it to the next layer. This:
- Stabilizes training by keeping activations in a well-behaved range
- Allows higher learning rates
- Acts as a mild regularizer by adding noise through the batch statistics
- Makes the network less sensitive to initialization

Placed after Dense, before activation — this is the standard order.

### `Dropout(0.3)`
During training, randomly zeros 30% of the neurons in the previous layer on each forward pass. This prevents neurons from co-adapting (becoming dependent on specific other neurons), which is the main cause of overfitting in neural networks. 0.3 is a moderate rate — aggressive enough to regularize but not so high that the model can't learn. It's placed after the first (widest) Dense layer where the capacity is highest and overfitting risk is greatest.

Dropout is disabled automatically at inference time (`.predict()`).

### `softmax` output activation
Converts the final layer's raw scores (logits) into a probability distribution over the 4 traffic classes. All outputs sum to 1, making them interpretable as class probabilities. Required for multi-class classification.

### `sparse_categorical_crossentropy` loss
Standard cross-entropy loss for multi-class problems. The "sparse" part means it accepts integer labels (0, 1, 2, 3) directly instead of one-hot encoded vectors. More memory efficient and avoids having to manually one-hot encode the labels.

### `adam` optimizer
Adaptive learning rate optimizer that maintains per-parameter learning rates. Works well on most problems without tuning. The default learning rate (0.001) is a good starting point for tabular data with this architecture.

### `EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)`
Stops training as soon as validation loss stops improving for 10 consecutive epochs. Without this, setting `epochs=100` as the cap would either underfit (if 100 is too few) or overfit (if the model peaks early but keeps training). `restore_best_weights=True` rolls back the model weights to the epoch with the best validation loss, not the final epoch — so you get the best generalization, not the most-trained version.

`validation_split=0.1` carves 10% of the training data as a held-out validation set used only by EarlyStopping (not by the test set evaluation).

---

## 6. Load & Prepare Dataset

### Fitting the scaler on training data only
```python
scaler = StandardScaler().fit(X_train)
```
The scaler learns the mean and standard deviation from the training set only. Applying it to test data (including noisy test data) uses those same training statistics. This is critical — fitting on the full dataset before splitting would leak test set statistics into the model, artificially inflating performance.

### `LabelEncoder` on the target
Converts `["Low", "Normal", "High", "Heavy"]` to integers `[0, 1, 2, 3]` (alphabetically sorted by default). All sklearn models and TensorFlow's `sparse_categorical_crossentropy` require integer or one-hot targets. The encoder is saved as an artifact so predictions can be decoded back to human-readable labels at deployment.

---

## 7. Model Selection

Five models are trained to span a spectrum of complexity and approach.

### Logistic Regression
The linear baseline. If traffic situation is linearly separable from the features (time of day, vehicle counts, day of week), logistic regression will match or beat everything else with far less compute.

- `solver="saga"`: supports both L1 and L2 regularization, handles larger datasets efficiently, required for `class_weight="balanced"`
- `max_iter=2000`: logistic regression with many features and imbalanced classes can converge slowly; the default 100 iterations is often not enough
- `class_weight="balanced"`: adjusts sample weights so that minority classes contribute more to the loss. Without it, the model can ignore rare classes entirely and still get high accuracy

### Random Forest
A non-linear ensemble that builds many decorrelated decision trees and averages their predictions. Handles feature interactions that logistic regression cannot.

- `n_estimators=300`: enough trees that the variance in the ensemble average is low. More trees always helps (up to a point of diminishing returns), 300 is a solid default
- `class_weight="balanced"`: same reason as above
- `n_jobs=-1`: use all available CPU cores for parallelizing tree construction

### Extra Trees (Extremely Randomized Trees)
Similar to Random Forest but uses random split thresholds rather than the optimal threshold. This extra randomization reduces variance further, often at the cost of a small bias increase. Frequently matches or beats Random Forest while being faster to train.

- `n_estimators=600`: individual ExtraTrees are noisier (more random), so you need more of them to get a stable ensemble average

### HistGradientBoosting
Scikit-learn's histogram-based gradient boosting (conceptually similar to LightGBM). Builds trees sequentially, each correcting the errors of the previous. Very fast due to binning continuous features into histograms.

- `max_depth=6`: limits individual tree depth to control overfitting. Gradient boosting is more prone to overfitting than bagging methods
- No `class_weight`: HistGradientBoosting handles class imbalance through the gradient weighting inherent in the boosting process; its class imbalance treatment is different from the bagging methods

### Neural Network
A fully-connected feed-forward network as the deep learning representative. On tabular data, neural networks often do not outperform well-tuned gradient boosting, but including one tests whether the non-linear capacity of deep learning adds anything on this specific problem, and allows comparison of noise robustness characteristics.

---

## 8. Model Testing

Each model is tested twice: once on the clean test set, once on the noisy test set. This produces two sets of metrics per model — the performance under ideal conditions and the performance under simulated sensor failure.

### Five metrics and why each one
**Accuracy** — the fraction of correct predictions. Included for completeness and familiarity, but unreliable on imbalanced class distributions.

**Balanced Accuracy** — the mean of per-class recall. A model that ignores a rare class entirely gets 0% balanced accuracy for that class, which pulls down the average. Unlike plain accuracy, it cannot be gamed by predicting only the majority class.

**F1 Macro** — unweighted mean of per-class F1 scores. F1 is the harmonic mean of precision and recall for each class. The "macro" average weights all classes equally regardless of size. This is the primary metric for comparing robustness because it penalizes poor performance on any class equally.

**Precision Macro** — of everything a model labels as class X, what fraction actually was X. High precision means few false alarms.

**Recall Macro** — of everything that actually was class X, what fraction did the model catch. High recall means few misses.

Both precision and recall are tracked because they trade off against each other — a model can game one at the expense of the other. Reporting both gives a complete picture.

---

## 9. Results & Robustness Decay

### The decay plot
```python
pivot["Decay"] = pivot["Clean"] - pivot["Noisy"]
```
For each model, `Decay` is the drop in F1 Macro when moving from clean to noisy test data. A decay of 0 means the model is completely unaffected by sensor noise. A large decay means the model is brittle.

This is the core research result: not which model is most accurate, but which model loses the least accuracy when real-world sensor noise is introduced. A slightly less accurate but far more robust model is more valuable in deployment.

The chart is sorted by decay ascending so the most robust model appears at the top.

---

## 10. Saved Artifacts

### `full_experiment_results.csv`
The complete metrics table for all models × conditions. Saved so results can be analyzed further (e.g., statistical tests, additional plots) without rerunning the entire experiment.

### `traffic_scaler.pkl` and `traffic_encoder.pkl`
The fitted scaler and label encoder must be saved alongside any model artifact. At deployment time, incoming sensor data must be normalized using the exact same mean/std the model was trained on, and predictions must be decoded using the exact same class-to-integer mapping. Recomputing these from new data would produce different values and break the model.

### `traffic_nn_model.keras`
Saved in Keras's native `.keras` format rather than the older `.h5`. The `.keras` format stores the full model (architecture + weights + optimizer state) in a single file and handles custom layers and configs more reliably.

### `joblib.dump` for sklearn objects
`joblib` serializes Python objects efficiently, with special handling for large NumPy arrays (memory-mapped files, compression). It is the recommended way to persist sklearn models and transformers, faster and more compact than `pickle` for objects containing arrays.
