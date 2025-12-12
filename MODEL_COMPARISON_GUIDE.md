# Model Comparison Guide for Imbalanced Traffic Congestion Data

## The Class Imbalance Problem

Your dataset has severe class imbalance:
- **Free flowing**: ~71% (14,361 samples)
- **Moderate delay**: ~11% (2,307 samples)
- **Light delay**: ~9% (1,897 samples)
- **Heavy delay**: ~9% (1,787 samples)

This 7:1 ratio between majority and minority classes requires special handling.

## Why Standard Models Fail on Imbalanced Data

1. **Bias toward majority class**: Models optimize overall accuracy, so they learn to predict "free flowing" for everything
2. **Poor minority class recall**: Rare congestion events get missed
3. **Misleading accuracy**: 71% accuracy by always predicting "free flowing"
4. **Business impact**: Missing actual congestion is worse than false alarms

## Models Compared in the Notebook

### 1. Random Forest (Balanced)
**Strategy**: Class weights
- Sets `class_weight='balanced'`
- Automatically adjusts weights inversely proportional to class frequencies
- Penalty for misclassifying minority classes is higher

**Pros**:
- Simple to implement
- Works with any sklearn classifier
- No data resampling needed

**Cons**:
- May still favor majority class
- Can lead to overfitting on minority classes

### 2. Balanced Random Forest
**Strategy**: Undersampling + ensemble
- From `imbalanced-learn` library
- Each tree sees balanced bootstrap sample
- Undersamples majority class for each estimator

**Pros**:
- Purpose-built for imbalanced data
- Maintains diversity through bootstrapping
- Often best performance

**Cons**:
- Discards majority class data
- Slower training

### 3. XGBoost (Scale Weight)
**Strategy**: Gradient boosting + weighting
- Uses `scale_pos_weight` parameter
- Sequential learning focuses on hard examples
- Built-in regularization

**Pros**:
- State-of-the-art performance
- Handles missing values
- Fast inference
- Good with minority classes

**Cons**:
- More hyperparameters to tune
- Can overfit if not careful

### 4. LightGBM (Balanced)
**Strategy**: Gradient boosting + class weights
- Uses `class_weight='balanced'`
- Leaf-wise tree growth (vs level-wise)
- Very fast training

**Pros**:
- Fastest training
- Memory efficient
- Often matches XGBoost performance
- Handles categorical features natively

**Cons**:
- Can overfit on small datasets
- Leaf-wise growth more sensitive to noise

### 5. Balanced Bagging
**Strategy**: Undersampling + bagging
- From `imbalanced-learn`
- Creates balanced bootstrap samples
- Trains multiple decision trees

**Pros**:
- Reduces variance
- Handles imbalance well
- Interpretable base estimators

**Cons**:
- Simpler than RF/GBM
- May need many estimators

## Metrics Used for Comparison

### F1 Score (Macro) - PRIMARY METRIC
- Average of per-class F1 scores
- Treats all classes equally (good for imbalanced data)
- Range: 0-1 (higher is better)
- **Use this to select best model**

### F1 Score (Weighted)
- Weighted average by class support
- Accounts for class imbalance in evaluation
- Better represents overall performance

### Balanced Accuracy
- Average of per-class recalls
- Not biased by class imbalance
- Good complement to F1 Macro

### Accuracy
- Overall correct predictions
- **DON'T use for model selection** (misleading on imbalanced data)
- Include for reference only

### Per-Class F1 Scores
- Shows performance on each congestion level
- Critical for understanding minority class performance
- Look for balanced performance across all classes

## How to Interpret Results

### Good Model Characteristics:
✅ **High F1 Macro** (>0.60 is good, >0.70 is excellent)
✅ **Balanced per-class F1** (all classes >0.50)
✅ **Good minority class recall** (catches actual congestion)
✅ **Reasonable training time** (<5 minutes for rapid iteration)

### Warning Signs:
⚠️ **High accuracy, low F1 Macro** (predicting majority class only)
⚠️ **Zero F1 for minority classes** (missing congestion entirely)
⚠️ **Huge gaps in per-class F1** (e.g., 0.95 vs 0.10)
⚠️ **Perfect training, poor validation** (overfitting)

## Expected Results

Based on similar imbalanced traffic datasets:

| Model | Expected F1 Macro | Best For |
|-------|------------------|----------|
| Balanced Random Forest | 0.65-0.75 | Overall best balance |
| XGBoost | 0.63-0.73 | Speed + performance |
| LightGBM | 0.64-0.74 | Large datasets, speed |
| Random Forest (Balanced) | 0.60-0.70 | Baseline |
| Balanced Bagging | 0.58-0.68 | Interpretability |

**Note**: Your actual results may vary based on:
- Feature quality (tracking accuracy)
- Feature engineering
- Hyperparameter tuning
- Data quality

## Hyperparameter Tuning Recommendations

### If Balanced Random Forest wins:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [2, 4, 6],
    'sampling_strategy': ['not majority', 'all']
}

grid_search = GridSearchCV(
    BalancedRandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)
```

### If XGBoost wins:
```python
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'min_child_weight': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'scale_pos_weight': [2, 3, 5]
}

grid_search = GridSearchCV(
    XGBClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)
```

### If LightGBM wins:
```python
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.03, 0.05, 0.1],
    'num_leaves': [31, 63, 127],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)
```

## Advanced Techniques (If Time Permits)

### 1. SMOTE + Ensemble
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])
```

### 2. Ensemble of Best Models
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier([
    ('brf', BalancedRandomForestClassifier(...)),
    ('xgb', XGBClassifier(...)),
    ('lgbm', LGBMClassifier(...))
], voting='soft', weights=[2, 1, 1])
```

### 3. Calibrated Classifier
```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(
    best_model,
    cv=5,
    method='isotonic'
)
```

### 4. Threshold Tuning
```python
# Get probabilities instead of predictions
y_proba = best_model.predict_proba(X_val)

# Try different thresholds for each class
from sklearn.metrics import f1_score
thresholds = np.arange(0.1, 0.9, 0.05)
best_threshold = {}

for class_idx in range(len(le.classes_)):
    best_f1 = 0
    for thresh in thresholds:
        y_pred_thresh = (y_proba[:, class_idx] >= thresh).astype(int)
        f1 = f1_score(y_val == class_idx, y_pred_thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold[class_idx] = thresh
```

## Troubleshooting

### All models predict only "free flowing"
**Cause**: Class weights too weak or features not predictive

**Solutions**:
1. Increase class weights manually
2. Try SMOTE oversampling
3. Check feature correlation with target
4. Add more discriminative features

### Poor performance on specific congestion level
**Cause**: Insufficient training samples or feature confusion

**Solutions**:
1. Check if "light" vs "moderate" delays are separable in feature space
2. Consider merging similar classes
3. Collect more examples of that class
4. Add class-specific features

### Models take too long to train
**Cause**: Too many estimators or large dataset

**Solutions**:
1. Reduce n_estimators (start with 100)
2. Use LightGBM instead of XGBoost
3. Sample training data for quick experiments
4. Use `max_samples` parameter in RF

### Overfitting (great training, poor validation)
**Cause**: Model too complex for data

**Solutions**:
1. Increase min_samples_split / min_samples_leaf
2. Reduce max_depth
3. Add regularization (L1/L2)
4. Use more cross-validation folds

## Final Model Selection Checklist

- [ ] Compared at least 3 different model types
- [ ] Evaluated using F1 Macro (not accuracy)
- [ ] Checked per-class performance (all >0.50 F1)
- [ ] Inspected confusion matrix for patterns
- [ ] Validated on held-out test set
- [ ] Reviewed feature importance
- [ ] Considered ensemble if close scores
- [ ] Documented hyperparameters
- [ ] Saved best model for submission

## References

- [Imbalanced-learn documentation](https://imbalanced-learn.org/)
- [XGBoost for imbalanced data](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)
- [LightGBM class weight](https://lightgbm.readthedocs.io/en/latest/Parameters.html#class_weight)
- [sklearn class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
