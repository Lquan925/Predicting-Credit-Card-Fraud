import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 1. Đọc dữ liệu
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

target_col = 'is_fraud'
id_col = 'id'
features = [c for c in train.columns if c not in [id_col, target_col]]

cat_cols = ['merchant_category', 'card_type', 'auth_method', 'channel', 'device_type']
bool_cols = [
    'is_foreign_transaction', 'is_new_merchant', 'used_vpn',
    'ip_country_mismatch', 'billing_shipping_mismatch', 'is_ai_generated_scam_attempt'
]

# 2. Tiền xử lý chuẩn kiểu dữ liệu
for col in cat_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

for col in bool_cols:
    train[col] = train[col].astype('int8')
    test[col] = test[col].astype('int8')

X = train[features]
y = train[target_col].astype('int8')
X_test = test[features]

# 3. Thiết lập Stratified 5-Fold CV & Train LightGBM Baseline
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
feature_importances = np.zeros(len(features))

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(**params, n_estimators=2000)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Lưu ý: Luôn dùng predict_proba[:, 1] cho ROC-AUC
    val_pred = model.predict_proba(X_va)[:, 1]
    oof_preds[val_idx] = val_pred
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
    feature_importances += model.feature_importances_ / skf.n_splits
    
    fold_auc = roc_auc_score(y_va, val_pred)
    print(f"Fold {fold + 1} ROC-AUC: {fold_auc:.6f} (Best iter: {model.best_iteration_})")

cv_auc = roc_auc_score(y, oof_preds)
print(f"\n>>> OVERALL OOF ROC-AUC (BASELINE): {cv_auc:.6f} <<<")

# 4. Xem Top 10 đặc trưng quan trọng nhất
imp_df = pd.DataFrame({'feature': features, 'importance': feature_importances}).sort_values('importance', ascending=False)
print("\nTop 10 Features:")
print(imp_df.head(10))

# 5. Xuất file nộp bài đầu tiên
sub = pd.DataFrame({
    'id': test[id_col],
    'is_fraud': test_preds
})
sub.to_csv('submission_baseline.csv', index=False)
print("\nĐã lưu file submission_baseline.csv thành công!")