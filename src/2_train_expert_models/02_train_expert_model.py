# ultimate_train_expert_model.py (FINAL CORRECTED VERSION)
import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import shap

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss, classification_report, confusion_matrix

# =================================================================================
# --- [ 중요! ] 분석 전 설정 ---
# =================================================================================
MODEL_TO_RUN = 'lgbm' 
EXPERT_NAME = 'mirna' 
# =================================================================================

# --- 스크립트 설정 ---
INPUT_FILENAME = f"{EXPERT_NAME}_multiclass_basemodel_data.csv"
DESKTOP_PATH = r"C:\Users\401-1\Desktop"
OUTPUT_DIR_NAME = f"{EXPERT_NAME.capitalize()}_pkl"
OUTPUT_DIR = os.path.join(DESKTOP_PATH, OUTPUT_DIR_NAME)
N_ESTIMATORS = 200
# --------------------

# --- 1. 모델 정의 ---
models = {
    'lgbm': lgb.LGBMClassifier(objective='multiclass', random_state=42, verbose=-1, n_estimators=N_ESTIMATORS),
    'rf': RandomForestClassifier(random_state=42, n_estimators=N_ESTIMATORS, n_jobs=-1, verbose=0),
    'xgb': xgb.XGBClassifier(objective='multi:softprob', random_state=42, n_estimators=N_ESTIMATORS, use_label_encoder=False, eval_metric='mlogloss'),
    'lr': LogisticRegression(random_state=42, max_iter=1000)
}

# --- 0. 스크립트 시작 및 설정 확인 ---
try:
    model = models[MODEL_TO_RUN]
    print(f"--- [{EXPERT_NAME.upper()}] 전문가 모델 종합 분석 시작 ---")
    print(f"--- 선택된 모델: {MODEL_TO_RUN.upper()} ---")
except KeyError:
    print(f"오류: '{MODEL_TO_RUN}'은 유효한 모델이 아닙니다. 'lgbm', 'rf', 'xgb', 'lr' 중에서 선택하세요.")
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"결과 파일은 '{OUTPUT_DIR}' 폴더에 저장됩니다.")

# --- 2. 데이터 로드 및 전처리 ---
input_path = os.path.join(DESKTOP_PATH, INPUT_FILENAME)
try:
    df = pd.read_csv(input_path, index_col='patient_barcode')
except FileNotFoundError:
    print(f"오류: 입력 파일 '{input_path}'을 찾을 수 없습니다. 해당 데이터의 통합 스크립트를 먼저 실행해주세요.")
    exit()

X = df.drop('label', axis=1)
y_str = df['label']

le = LabelEncoder()
y = le.fit_transform(y_str)
num_classes = len(le.classes_)
print(f"학습할 암종: {le.classes_} ({num_classes}개 클래스)")
joblib.dump(le, os.path.join(OUTPUT_DIR, f'{EXPERT_NAME}_label_encoder.pkl'))


# --- 3. 교차 검증 및 모델 학습/평가 ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(df), num_classes))
val_scores = []
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# --- [수정] SHAP 분석을 위한 리스트 초기화 ---
all_shap_values = []
all_X_vals = [] # <-- 이 줄이 누락되어 오류가 발생했습니다.
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

print("\n5-겹 교차 검증 방식으로 모델 학습을 시작합니다...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"\n--- Fold {fold+1} 학습 및 예측 시작 ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # 모델 학습
    if MODEL_TO_RUN in ['lgbm', 'xgb']:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
    else:
        model.fit(X_train, y_train)

    # 학습 곡선 시각화 (LGBM만 지원)
    if MODEL_TO_RUN == 'lgbm':
        plt.figure(figsize=(10, 5))
        lgb.plot_metric(model, metric='multi_logloss', title=f'{EXPERT_NAME} Fold {fold+1} 학습 곡선')
        plt.savefig(os.path.join(OUTPUT_DIR, f'{EXPERT_NAME}_{MODEL_TO_RUN}_learning_curve_fold_{fold+1}.png'))
        plt.close()

    # 상세 성능 지표 출력
    train_pred_proba = model.predict_proba(X_train)
    train_pred_label = np.argmax(train_pred_proba, axis=1)
    train_acc = accuracy_score(y_train, train_pred_label)
    train_f1 = f1_score(y_train, train_pred_label, average='weighted')
    train_loss = log_loss(y_train, train_pred_proba)
    print(f"  [Train] Acc: {train_acc:.4f}, F1: {train_f1:.4f}, LogLoss: {train_loss:.4f}")

    val_pred_proba = model.predict_proba(X_val)
    val_pred_label = np.argmax(val_pred_proba, axis=1)
    val_acc = accuracy_score(y_val, val_pred_label)
    val_f1 = f1_score(y_val, val_pred_label, average='weighted')
    val_loss = log_loss(y_val, val_pred_proba)
    val_auc = roc_auc_score(y_val, val_pred_proba, multi_class='ovr')
    print(f"  [Valid] Acc: {val_acc:.4f}, F1: {val_f1:.4f}, LogLoss: {val_loss:.4f}, AUC: {val_auc:.4f}")
    
    oof_preds[val_idx] = val_pred_proba
    val_scores.append({'acc': val_acc, 'f1': val_f1, 'auc': val_auc})
    
    # XAI (SHAP) 분석
    print("  - SHAP 값 계산 중...")
    if MODEL_TO_RUN in ['lgbm', 'rf', 'xgb']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)
    else: 
        X_val_sample = shap.sample(X_val, 10) 
        explainer = shap.KernelExplainer(model.predict_proba, X_val_sample)
        shap_values = explainer.shap_values(X_val)

    all_shap_values.append(shap_values)
    all_X_vals.append(X_val)
    
    # Fold별 모델 저장
    model_path = os.path.join(OUTPUT_DIR, f'{EXPERT_NAME}_{MODEL_TO_RUN}_model_fold_{fold+1}.pkl')
    joblib.dump(model, model_path)

# --- 4. 최종 결과 종합 및 시각화 ---
print("\n--- 교차 검증 최종 요약 ---")
avg_val_acc = np.mean([s['acc'] for s in val_scores])
avg_val_f1 = np.mean([s['f1'] for s in val_scores])
avg_val_auc = np.mean([s['auc'] for s in val_scores])
print(f"평균 검증(Validation) Accuracy: {avg_val_acc:.4f}")
print(f"평균 검증(Validation) F1-Score: {avg_val_f1:.4f}")
print(f"평균 검증(Validation) AUC: {avg_val_auc:.4f}")

overall_oof_auc = roc_auc_score(y, oof_preds, multi_class='ovr')
print(f"\n{EXPERT_NAME.capitalize()} 모델의 전체 OOF AUC: {overall_oof_auc:.4f}")

# 최종 분류 리포트 및 혼동 행렬
print("\n--- 전체 OOF 예측 최종 상세 분석 ---")
oof_pred_labels = np.argmax(oof_preds, axis=1)
report = classification_report(y, oof_pred_labels, target_names=le.classes_)
print("\n[최종 분류 리포트]")
print(report)

print("\n[최종 혼동 행렬] 그래프를 생성하고 저장합니다...")
cm = confusion_matrix(y, oof_pred_labels)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title(f'[{EXPERT_NAME.upper()}/{MODEL_TO_RUN.upper()}] 최종 혼동 행렬 (OOF 예측)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass
cm_fig_path = os.path.join(OUTPUT_DIR, f'{EXPERT_NAME}_{MODEL_TO_RUN}_confusion_matrix.png')
plt.savefig(cm_fig_path)
print(f"혼동 행렬 그래프가 '{cm_fig_path}'에 저장되었습니다.")
plt.close()


# --- SHAP 값 종합 및 특징 중요도 시각화 (v2: 버그 수정) ---
print("\n--- SHAP 기반 특징 중요도 분석 ---")
try:
    X_val_all = pd.concat(all_X_vals)
    # SHAP 값의 형태(list 또는 array)에 따라 유연하게 통합
    if isinstance(all_shap_values[0], list):
        shap_values_combined = [np.concatenate([s[i] for s in all_shap_values], axis=0) for i in range(num_classes)]
    else:
        shap_values_combined = np.concatenate(all_shap_values, axis=0)

    # 1. 종합 특징 중요도 (Top 20)
    shap.summary_plot(shap_values_combined, X_val_all, plot_type="bar",
                      class_names=le.classes_, show=False, max_display=20)
    plt.title(f"[{EXPERT_NAME.upper()}/{MODEL_TO_RUN.upper()}] 종합 특징 중요도 (Top 20)")
    plt.savefig(os.path.join(OUTPUT_DIR, f'{EXPERT_NAME}_{MODEL_TO_RUN}_1_overall_feature_importance.png'), bbox_inches='tight')
    plt.close()
    print("종합 특징 중요도 그래프 저장 완료.")

    # 2. 오믹스 종류별 중요도
    print("\n[오믹스 종류별 중요도 분석]")
    
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # --- [수정] 중요도 점수 계산 로직 강화 ---
    if isinstance(shap_values_combined, list):
        # shap_values가 클래스별 list일 경우, (class, sample, feature)의 3D 배열로 변환
        shap_values_np = np.array(shap_values_combined)
        # 클래스와 샘플 축 모두에 대해 절대값의 평균을 내어 특징별 중요도 계산
        importance_scores = np.mean(np.mean(np.abs(shap_values_np), axis=1), axis=0)
    else: # 이미 (sample, feature, class)의 3D 배열인 경우 (예: RandomForest)
        # 클래스와 샘플 축 모두에 대해 절대값의 평균을 냄
        importance_scores = np.mean(np.mean(np.abs(shap_values_combined), axis=2), axis=0)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    feature_importance_df = pd.DataFrame({'feature': X_val_all.columns, 'importance': importance_scores})
    
    if feature_importance_df['feature'].str.contains('_').any():
        feature_importance_df['omics_type'] = feature_importance_df['feature'].str.split('_').str[-1]
        
        if feature_importance_df['omics_type'].nunique() > 1:
            omics_importance = feature_importance_df.groupby('omics_type')['importance'].sum().sort_values(ascending=False)
            
            plt.figure(figsize=(10, 6))
            omics_importance.plot(kind='bar', color='skyblue')
            plt.title(f'[{EXPERT_NAME.upper()}/{MODEL_TO_RUN.upper()}] 오믹스 종류별 특징 중요도')
            plt.ylabel('Total SHAP Value')
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(OUTPUT_DIR, f'{EXPERT_NAME}_{MODEL_TO_RUN}_2_omics_type_importance.png'), bbox_inches='tight')
            plt.close()
            print("오믹스별 중요도 그래프 저장 완료.")

            print("\n[각 오믹스 내 중요 특징 Top 10]")
            top_features_per_omics = feature_importance_df.groupby('omics_type').apply(lambda x: x.nlargest(10, 'importance')).reset_index(drop=True)
            print(top_features_per_omics[['omics_type', 'feature', 'importance']])
        else:
            print("고유한 오믹스 종류가 1개뿐이므로, 오믹스별 분석을 건너뜁니다.")
    else:
        print("특징 이름에 오믹스 종류를 구분할 접미사('_')가 없으므로, 오믹스별 분석을 건너뜁니다.")

except Exception as e:
    print(f"SHAP 기반 분석 중 예상치 못한 오류 발생: {e}")

# --- 5. 최종 결과물 저장 ---
pred_cols = [f'pred_{EXPERT_NAME}_{cls}' for cls in le.classes_]
oof_df = pd.DataFrame(oof_preds, index=df.index, columns=pred_cols)
oof_df.reset_index(inplace=True)
oof_output_path = os.path.join(DESKTOP_PATH, f'oof_preds_{EXPERT_NAME}.csv')
oof_df.to_csv(oof_output_path, index=False)
print(f"\nOOF 예측 파일이 바탕화면에 '{os.path.basename(oof_output_path)}' 이름으로 저장되었습니다.")

print("\n--- 모든 작업 완료 ---")