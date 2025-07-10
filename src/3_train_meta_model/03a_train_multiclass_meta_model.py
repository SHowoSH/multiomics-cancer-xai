# 3_train_meta_model_final.py (v3.1: SHAP 분석 최종 수정)
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import shap
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import json # json 모듈 추가

warnings.filterwarnings('ignore', category=UserWarning)

print("--- [최종 단계] 메타 모델 상세 분석 시작 (XAI 포함 v3.1) ---")

# --- 설정 ---
# 모든 입력/출력 파일의 기준 경로 (바탕화면)
DESKTOP_PATH = r"C:\Users\401-1\Desktop" # 로컬에서 실행 시 사용하는 경로
# 서버 업로드 경로를 가정하여 OUTPUT_DIR 설정
# 서버에서는 ml_models/cancer_type_classification/Meta_analysis_pkl/ 이 경로가 될 것임
# 로컬에서 이 스크립트를 실행하여 모델을 생성할 때, 해당 OUTPUT_DIR에 파일들이 생성됩니다.
# 생성 후 이 파일들을 서버의 ml_models/cancer_type_classification/Meta_analysis_pkl/ 로 이동해야 합니다.
OUTPUT_DIR = os.path.join(DESKTOP_PATH, "Meta_analysis_pkl") # 로컬 실행 시 임시 출력 폴더

# 각 오믹스별 기본 모델의 OOF (Out-Of-Fold) 예측 파일 목록
# 이 파일들은 DESKTOP_PATH에 직접 위치해야 합니다.
OOF_PREDS_FILES = [
    'oof_preds_gene.csv',
    'oof_preds_meth.csv',
    'oof_preds_mutation.csv',
    'oof_preds_cnv.csv',
    'oof_preds_mirna.csv'
]

# LabelEncoder 객체 파일 경로
# 이 파일은 DESKTOP_PATH/Gene_pkl/ 폴더 내에 있어야 합니다.
# 이 LabelEncoder는 개별 Expert 모델용이며, 메타 모델의 최종 레이블을 위한 것이 아닙니다.
# 메타 모델의 레이블 인코더는 이 스크립트 내에서 직접 학습되고 저장됩니다.
LABEL_ENCODER_PATH = os.path.join(DESKTOP_PATH, 'Gene_pkl', 'gene_label_encoder.pkl')

# 레이블 정보를 가져올 원본 데이터 파일 경로 (patient_barcode와 label 컬럼 포함)
# 이 파일은 DESKTOP_PATH에 직접 위치해야 합니다.
LABEL_SOURCE_FILE = os.path.join(DESKTOP_PATH, 'gene_multiclass_basemodel_data.csv')


os.makedirs(OUTPUT_DIR, exist_ok=True) # 폴더가 없으면 생성

print(f"\n설정된 기본 경로: {DESKTOP_PATH}")
print(f"OOF 예측 파일 목록: {OOF_PREDS_FILES}")
print(f"Label Encoder 경로: {LABEL_ENCODER_PATH}")
print(f"레이블 소스 파일: {LABEL_SOURCE_FILE}")
print(f"출력 디렉토리: {OUTPUT_DIR}")


# 1. 데이터 준비
meta_df = pd.DataFrame()
print("\n각 전문가 모델의 예측(OOF) 파일을 통합합니다...")
for file_name in OOF_PREDS_FILES:
    path = os.path.join(DESKTOP_PATH, file_name)
    try:
        df = pd.read_csv(path)
        # OOF 파일의 첫 번째 컬럼이 patient_barcode인지 확인하고 인덱스로 설정
        if 'patient_barcode' in df.columns:
            df = df.set_index('patient_barcode')
        else:
            # 첫 번째 컬럼을 patient_barcode로 간주
            df = df.set_index(df.columns[0])
            df.index.name = 'patient_barcode' # 인덱스 이름 설정

        if meta_df.empty:
            meta_df = df
        else:
            # label 컬럼은 label_df에서 가져오므로 OOF 파일에서 제거 (충돌 방지)
            meta_df = pd.merge(meta_df, df.drop(columns=['label'], errors='ignore'), on='patient_barcode', how='outer')
        print(f" '{file_name}' 통합 완료. 현재 메타 데이터프레임 크기: {meta_df.shape}")
    except FileNotFoundError:
        print(f"경고: '{path}' 파일을 찾을 수 없습니다. 이 파일은 건너뜝니다.")
    except Exception as e:
        print(f" '{file_name}' 통합 중 오류 발생: {e}. 이 파일은 건너웁니다.")


try:
    # Label Encoder 로드 (이것은 Expert 모델의 LabelEncoder이며, 최종 메타 모델의 LabelEncoder가 아님)
    # 이 le는 num_classes를 얻기 위해 사용됩니다.
    le_expert = joblib.load(LABEL_ENCODER_PATH)
    num_classes = len(le_expert.classes_)
    print(f"\nExpert 모델용 LabelEncoder 로드 완료. 학습된 클래스: {le_expert.classes_}, 개수: {num_classes}")

    # 레이블 정보 로드 및 통합 (메타 모델의 실제 최종 레이블)
    label_df = pd.read_csv(LABEL_SOURCE_FILE, usecols=['patient_barcode', 'label'])
    label_df = label_df.set_index('patient_barcode') # patient_barcode를 인덱스로 설정

    if 'label' in meta_df.columns:
        meta_df = meta_df.drop(columns=['label']) # 기존 label 컬럼이 있다면 제거

    meta_df = pd.merge(meta_df, label_df, on='patient_barcode', how='left')
    meta_df.dropna(subset=['label'], inplace=True) # 레이블이 없는 샘플 제거 (매우 중요)

    # 예측 컬럼 식별 및 결측치 처리
    pred_cols = [col for col in meta_df.columns if col.startswith('pred_')]
    if not pred_cols:
        print("오류: 예측 컬럼('pred_')을 찾을 수 없습니다. OOF 파일 형식을 확인하세요.")
        exit()
    meta_df[pred_cols] = meta_df[pred_cols].fillna(1/num_classes)

    print(f"데이터 준비 완료. 최종 메타 데이터프레임 크기: {meta_df.shape}")
    print("메타 데이터프레임의 레이블 분포:")
    print(meta_df['label'].value_counts())

except FileNotFoundError as e:
    print(f"오류: 필수 파일이 누락되었습니다. ({e})")
    print(f" '{LABEL_ENCODER_PATH}' 또는 '{LABEL_SOURCE_FILE}' 파일을 확인해주세요.")
    exit()
except Exception as e:
    print(f"오류: 데이터 준비 중 문제가 발생했습니다. (오류: {e})"); exit()


# 2. 메타 모델 학습 데이터 준비
X_meta = meta_df[pred_cols] # 최종 메타 모델의 입력은 'pred_'로 시작하는 예측 컬럼들
y_meta_str = meta_df['label']

# 메타 모델의 최종 LabelEncoder 생성 및 학습
le_meta = LabelEncoder()
y_meta = le_meta.fit_transform(y_meta_str)
print(f"메타 모델 학습 데이터 준비 완료. 특징 수: {X_meta.shape[1]}, 샘플 수: {X_meta.shape[0]}")
print(f"메타 모델이 학습할 최종 암종 클래스: {le_meta.classes_} ({len(le_meta.classes_)}개 클래스)")


# 3. 메타 모델 교차 검증
print("\n최종 메타 모델을 5-겹 교차 검증으로 학습 및 평가합니다...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_predictions = np.zeros_like(y_meta)
all_shap_values_folds = [] # 각 폴드에서 반환되는 3차원 SHAP 값을 그대로 저장
all_X_val_dfs = [] # 각 폴드의 X_val (DataFrame)을 저장

for fold, (train_idx, val_idx) in enumerate(cv.split(X_meta, y_meta)):
    X_train, X_val = X_meta.iloc[train_idx], X_meta.iloc[val_idx]
    y_train, y_val = y_meta[train_idx], y_meta[val_idx]

    meta_model = lgb.LGBMClassifier(objective='multiclass', num_class=num_classes,
                                    n_estimators=100, # OOF 예측값은 이미 과적합이 덜 되어있으므로, n_estimators를 더 낮춰볼 수 있음
                                    random_state=42, verbose=-1,
                                    learning_rate=0.05, num_leaves=5, max_depth=3, # 간단한 모델을 위해 제한
                                    reg_alpha=0.1, reg_lambda=0.1 # 간단한 정규화
                                   )
    meta_model.fit(X_train, y_train) # 간단한 메타모델은 early stopping 없이 전체 학습

    fold_preds = meta_model.predict(X_val)
    final_predictions[val_idx] = fold_preds

    # SHAP 값 계산
    explainer = shap.TreeExplainer(meta_model)
    shap_values_current_fold = explainer.shap_values(X_val)

    # --- SHAP 디버깅 출력 ---
    print(f"\nDEBUG SHAP (Fold {fold+1}): type(shap_values_current_fold): {type(shap_values_current_fold)}")
    if isinstance(shap_values_current_fold, np.ndarray) and shap_values_current_fold.ndim == 3:
        print(f"DEBUG SHAP (Fold {fold+1}): shap_values_current_fold shape: {shap_values_current_fold.shape} (Expected: (samples, {X_val.shape[1]}, {num_classes}))")
        if shap_values_current_fold.shape[1] != X_val.shape[1]:
            print(f"!!!!! WARNING SHAP: Feature count mismatch in current fold SHAP values ({shap_values_current_fold.shape[1]} vs {X_val.shape[1]}) !!!!!")
        if shap_values_current_fold.shape[2] != num_classes:
            print(f"!!!!! WARNING SHAP: Class count mismatch in current fold SHAP values ({shap_values_current_fold.shape[2]} vs {num_classes}) !!!!!")
    elif isinstance(shap_values_current_fold, list) and len(shap_values_current_fold) == num_classes:
        print(f"DEBUG SHAP (Fold {fold+1}): shap_values_current_fold is a list of arrays (older SHAP version).")
        for i, arr in enumerate(shap_values_current_fold):
            print(f"   shap_values_current_fold[{i}] shape: {arr.shape} (Expected: (samples, {X_val.shape[1]}))")
            if arr.shape[1] != X_val.shape[1]:
                print(f"   !!!!! WARNING SHAP: Feature count mismatch in inner array shap_values_current_fold[{i}] !!!!!")
    else:
        print(f"DEBUG SHAP (Fold {fold+1}): shap_values_current_fold has unexpected type/shape: {type(shap_values_current_fold)}, {getattr(shap_values_current_fold, 'shape', 'N/A')}")
    # --- SHAP 디버깅 출력 끝 ---

    all_shap_values_folds.append(shap_values_current_fold)
    all_X_val_dfs.append(X_val) # X_val (DataFrame)을 저장

# 4. 최종 성능 평가
print("\n=============================================")
print("===        앙상블 모델 최종 성능 평가         ===")
print("=============================================")
final_accuracy = accuracy_score(y_meta, final_predictions)
print(f"\n최종 앙상블 모델의 교차 검증 정확도: {final_accuracy * 100:.2f}%")
print("\n[최종 분류 리포트]")
print(classification_report(y_meta, final_predictions, target_names=le_meta.classes_)) # le_meta 사용
cm = confusion_matrix(y_meta, final_predictions)
cm_df = pd.DataFrame(cm, index=le_meta.classes_, columns=le_meta.classes_) # le_meta 사용
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('최종 앙상블 모델 혼동 행렬')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass
cm_fig_path = os.path.join(DESKTOP_PATH, 'final_ensemble_confusion_matrix.png')
plt.savefig(cm_fig_path)
print("최종 혼동 행렬 그래프가 바탕화면에 저장되었습니다.")
plt.close()


# --- 메타 모델 SHAP 기반 특징 중요도 분석 ---
print("\n--- 메타 모델 SHAP 기반 특징 중요도 분석 ---")
try:
    X_val_all_combined = pd.concat(all_X_val_dfs, ignore_index=False) # 원본 인덱스 유지

    # SHAP 값 통합 로직 (3차원 배열 또는 리스트 형태 모두 처리)
    if all_shap_values_folds and isinstance(all_shap_values_folds[0], np.ndarray) and all_shap_values_folds[0].ndim == 3:
        concatenated_all_shap_3d = np.concatenate(all_shap_values_folds, axis=0)
        shap_values_combined_list = [concatenated_all_shap_3d[:, :, c] for c in range(num_classes)]
        print(f"DEBUG SHAP Combined: 3D SHAP values integrated. Resulting list of arrays shape (first element): {shap_values_combined_list[0].shape}")
    elif all_shap_values_folds and isinstance(all_shap_values_folds[0], list) and len(all_shap_values_folds[0]) == num_classes:
        shap_values_combined_list = [np.concatenate([s[i] for s in all_shap_values_folds], axis=0) for i in range(num_classes)]
        print(f"DEBUG SHAP Combined: List of 2D SHAP values integrated. Resulting list of arrays shape (first element): {shap_values_combined_list[0].shape}")
    else:
        print("ERROR: SHAP 값의 반환 형태가 예상과 다릅니다. SHAP 분석을 수행할 수 없습니다.")
        shap_values_combined_list = [] # 비어있는 리스트로 설정하여 하위 오류 방지


    if shap_values_combined_list and shap_values_combined_list[0].shape[0] == X_val_all_combined.shape[0] and shap_values_combined_list[0].shape[1] == X_val_all_combined.shape[1]:
        # 1. 종합 특징 중요도 (Top 20) - 어떤 '전문가 의견'이 중요했나
        shap.summary_plot(shap_values_combined_list, X_val_all_combined, plot_type="bar", class_names=le_meta.classes_, show=False, max_display=20) # le_meta 사용
        plt.title("[메타 모델] 종합 특징 중요도 (Top 20)")
        plt.savefig(os.path.join(OUTPUT_DIR, 'meta_model_1_overall_feature_importance.png'), bbox_inches='tight')
        plt.close()
        print("종합 특징 중요도 그래프 저장 완료.")

        # 2. 오믹스 종류별 중요도 - 어떤 '전문가'가 중요했나
        shap_values_np = np.array(shap_values_combined_list) # (num_classes, total_samples, features)
        importance_scores = np.mean(np.mean(np.abs(shap_values_np), axis=1), axis=0) # (features,)

        feature_importance_df = pd.DataFrame({'feature': X_val_all_combined.columns, 'importance': importance_scores})

        # OOF 파일명에서 오믹스 타입 추출 (예: 'pred_gene_BRCA' -> 'gene')
        # pred_XXX 형태가 아니라면, 이 부분 수정이 필요
        feature_importance_df['omics_type'] = feature_importance_df['feature'].apply(
            lambda x: x.split('_')[1] if x.startswith('pred_') and len(x.split('_')) > 1 else 'other_omics' # 'pred_gene_BRCA' -> 'gene'
        )

        omics_importance = feature_importance_df.groupby('omics_type')['importance'].sum().sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        omics_importance.plot(kind='bar', color='skyblue')
        plt.title('[메타 모델] 전문가(오믹스)별 중요도 총합')
        plt.ylabel('Total SHAP Value')
        plt.xticks(rotation=0)
        plt.savefig(os.path.join(OUTPUT_DIR, 'meta_model_2_omics_type_importance.png'), bbox_inches='tight')
        plt.close()
        print("전문가(오믹스)별 중요도 그래프 저장 완료.")

        print("\n[각 오믹스 내 중요 특징 Top 10]")
        top_features_per_omics = feature_importance_df.groupby('omics_type').apply(lambda x: x.nlargest(10, 'importance')).reset_index(drop=True)
        print(top_features_per_omics[['omics_type', 'feature', 'importance']])

    else:
        print("SHAP 분석을 위한 데이터가 유효하지 않습니다. 그래프 생성 건너뜜.")

except Exception as e:
    print(f"SHAP 기반 분석 중 오류 발생: {e}")
# ----------------------------------------------------

# 5. 최종 모델 저장
print("\n전체 데이터로 최종 메타 모델을 재학습하고 저장합니다...")
# 최종 모델 학습 (cross-validation 외부)
final_meta_model = lgb.LGBMClassifier(objective='multiclass', num_class=num_classes, n_estimators=100, random_state=42, verbose=-1, learning_rate=0.05, num_leaves=5, max_depth=3, reg_alpha=0.1, reg_lambda=0.1)
final_meta_model.fit(X_meta, y_meta)

# --- [수정!] 메타 모델 관련 파일 저장 로직 추가 ---
# 메타 모델 저장
joblib.dump(final_meta_model, os.path.join(OUTPUT_DIR, 'final_meta_model.pkl'))
print("성공! 'final_meta_model.pkl'이 저장되었습니다.")

# 메타 모델 LabelEncoder 저장
joblib.dump(le_meta, os.path.join(OUTPUT_DIR, 'meta_label_encoder.pkl'))
print("성공! 'meta_label_encoder.pkl'이 저장되었습니다.")

# 메타 모델 Feature Names 저장
meta_feature_names = X_meta.columns.tolist()
with open(os.path.join(OUTPUT_DIR, 'meta_features_for_ensemble.json'), 'w') as f:
    json.dump(meta_feature_names, f)
print("성공! 'meta_features_for_ensemble.json'이 저장되었습니다.")

# 최종 임퓨터는 현재 이 스크립트에서 학습/사용되지 않으므로 저장 로직은 생략.
# 만약 필요하다면 메타 모델 학습 전 X_meta에 대한 임퓨터를 학습하고 저장하는 로직을 추가해야 함.

print("\n--- 모든 작업 완료 ---")