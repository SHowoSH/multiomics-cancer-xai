# B3_train_binary_meta_model_deluxe.py (v5: tqdm 임포트 및 메타 모델 하이퍼파라미터 튜닝)
import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import lightgbm as lgb
import shap
from tqdm import tqdm # <-- tqdm 임포트 추가

from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV # RandomizedSearchCV 임포트
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss, classification_report, confusion_matrix, make_scorer # make_scorer 임포트

# --- 설정 ---
MODEL_NAME_PREFIX = 'meta_binary' 
MODEL_TO_RUN = 'lgbm' 
DESKTOP_PATH = r"C:\Users\401-1\Desktop"
OUTPUT_DIR_NAME = f"{MODEL_NAME_PREFIX.capitalize()}_pkl"
OUTPUT_DIR = os.path.join(DESKTOP_PATH, OUTPUT_DIR_NAME)
N_ESTIMATORS = 500 # 메타 모델의 기본 n_estimators (튜닝 범위에서 더 높은 값 포함)
OOF_PREDS_FILES = [
    'oof_preds_gene_binary.csv', 
    'oof_preds_meth_binary.csv', 
    'oof_preds_mutation_binary.csv', # 이전에 제외했으나, 다시 포함시켜 10개 특징을 만들 때 사용합니다.
    'oof_preds_cnv_binary.csv', 
    'oof_preds_mirna_binary.csv'
]
LABEL_ENCODER_PATH = os.path.join(DESKTOP_PATH, 'Gene_binary_pkl', 'gene_binary_label_encoder.pkl')
LABEL_SOURCE_FILE = os.path.join(DESKTOP_PATH, 'gene_binary_basemodel_data.csv') 
# -----------

# --- 학습 곡선 시각화 함수 (직접 그리는 방식으로 변경) ---
def plot_learning_curve_custom(evals_result, fold, save_path, model_name_prefix, model_to_run):
    try:
        if evals_result and 'valid_0' in evals_result:
            metric_to_plot = None
            metric_name = ""
            if 'logloss' in evals_result['valid_0']:
                metric_to_plot = evals_result['valid_0']['logloss']
                metric_name = "LogLoss"
            elif 'auc' in evals_result['valid_0']: # logloss가 없으면 auc로 대체
                metric_to_plot = evals_result['valid_0']['auc']
                metric_name = "AUC"
                # AUC는 보통 높을수록 좋으므로, 반대로 뒤집어서 그리는 것을 고려할 수 있음
                # plt.plot(1 - np.array(metric_to_plot), label='Validation 1-AUC')

            if metric_to_plot is not None:
                plt.figure(figsize=(10, 5))
                plt.plot(metric_to_plot, label=f'Validation {metric_name}')
                plt.title(f'{model_name_prefix} Fold {fold+1} 학습 곡선 ({metric_name})')
                plt.xlabel('Boosting Round')
                plt.ylabel(metric_name)
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(save_path, f'{model_name_prefix}_{model_to_run}_learning_curve_fold_{fold+1}.png'))
                plt.close()
            else:
                print(f"경고: Fold {fold+1} 학습 곡선 그래프를 생성할 수 없습니다. evals_result_에 유효한 메트릭이 없습니다.")
        else:
            print(f"경고: Fold {fold+1} 학습 곡선 그래프를 생성할 수 없습니다. evals_result_가 비어 있습니다.")
    except Exception as e:
        print(f"Fold {fold+1} 학습 곡선 그래프 저장 실패: {e}")

# --- 0. 스크립트 시작 ---
print(f"--- [{MODEL_NAME_PREFIX.upper()}] 최종 앙상블 모델 상세 분석 시작 ---")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"결과 파일은 '{OUTPUT_DIR}' 폴더에 저장됩니다.")

# --- 1. 데이터 준비 ---
meta_df = pd.DataFrame()
print("\n각 전문가 모델의 예측(OOF) 파일을 통합합니다...")
for file_name in OOF_PREDS_FILES:
    path = os.path.join(DESKTOP_PATH, file_name)
    try:
        df_oof = pd.read_csv(path)
        # patient_barcode를 인덱스로 설정
        if 'patient_barcode' in df_oof.columns:
            df_oof = df_oof.set_index('patient_barcode')
        else: 
            df_oof = df_oof.set_index(df_oof.columns[0]) 
            df_oof.index.name = 'patient_barcode' 

        # OOF 파일에서 예측 컬럼 추출 및 새 컬럼명으로 변경 (10개 특징 생성 로직)
        omics_name = file_name.replace('oof_preds_', '').replace('_binary.csv', '')
        
        # 실제 파일 내용에 맞게 `pred_X_binary` 컬럼에서 Normal과 Cancer 확률을 추출
        if f'pred_{omics_name}_binary' in df_oof.columns:
            df_oof[f'pred_{omics_name}_normal_prob'] = df_oof[f'pred_{omics_name}_binary']
            df_oof[f'pred_{omics_name}_cancer_prob'] = 1 - df_oof[f'pred_{omics_name}_binary']
            df_oof = df_oof[[f'pred_{omics_name}_normal_prob', f'pred_{omics_name}_cancer_prob']] 
        # 만약 OOF 파일에 'pred_0' (Cancer)와 'pred_1' (Normal) 두 컬럼이 이미 있다면
        elif 'pred_0' in df_oof.columns and 'pred_1' in df_oof.columns:
            df_oof[f'pred_{omics_name}_cancer_prob'] = df_oof['pred_0']
            df_oof[f'pred_{omics_name}_normal_prob'] = df_oof['pred_1']
            df_oof = df_oof[[f'pred_{omics_name}_cancer_prob', f'pred_{omics_name}_normal_prob']]
        else:
            print(f"경고: '{file_name}'에서 예상된 예측 컬럼 형식을 찾을 수 없습니다. 이 파일을 건너뜜. (오믹스: {omics_name})")
            continue # 파일 건너뛰기

        if meta_df.empty:
            meta_df = df_oof
        else:
            meta_df = pd.merge(meta_df, df_oof, on='patient_barcode', how='outer')
        print(f" '{file_name}' 통합 완료. 현재 메타 데이터프레임 크기: {meta_df.shape}")
    except FileNotFoundError:
        print(f"경고: '{path}' 파일을 찾을 수 없습니다. 이 파일은 건너뜝니다.")
    except Exception as e:
        print(f" '{file_name}' 통합 중 오류 발생: {e}. 이 파일은 건너뜝니다.")

try:
    # Label Encoder 로드
    le = joblib.load(LABEL_ENCODER_PATH)
    if len(le.classes_) != 2:
        raise ValueError(f"LabelEncoder가 학습한 클래스 개수가 2개가 아닙니다: {le.classes_}. 이진 분류 모델에는 'Cancer'/'Normal' 2개 클래스만 필요합니다.")

    # 레이블 정보 로드 및 통합
    label_df = pd.read_csv(LABEL_SOURCE_FILE, usecols=['patient_barcode', 'label'])
    label_df = label_df.set_index('patient_barcode') 

    if 'label' in meta_df.columns: 
        meta_df = meta_df.drop(columns=['label']) 

    meta_df = pd.merge(meta_df, label_df, on='patient_barcode', how='left')
    meta_df.dropna(subset=['label'], inplace=True) # 레이블이 없는 샘플 제거
    
    pred_cols = [col for col in meta_df.columns if col.startswith('pred_')]
    if not pred_cols:
        print("오류: 예측 컬럼('pred_')을 찾을 수 없습니다. OOF 파일 형식을 확인하거나 통합 과정에서 문제가 발생했습니다.")
        exit()
    meta_df[pred_cols] = meta_df[pred_cols].fillna(0.5) 
    
    print(f"데이터 준비 완료. 최종 메타 데이터프레임 크기: {meta_df.shape}")
    print("메타 데이터프레임의 레이블 분포:")
    print(meta_df['label'].value_counts())

except FileNotFoundError as e:
    print(f"오류: 필수 파일이 누락되었습니다. ({e})")
    print(f" '{LABEL_ENCODER_PATH}' 또는 '{LABEL_SOURCE_FILE}' 파일을 확인해주세요.")
    exit()
except Exception as e:
    print(f"오류: 데이터 준비 중 문제가 발생했습니다. (오류: {e})"); exit()

# --- 2. 메타 모델 학습 데이터 준비 ---
X_meta = meta_df.drop(columns=['patient_barcode', 'label'], errors='ignore') 
y_meta_str = meta_df['label']
y_meta = le.transform(y_meta_str)

# LabelEncoder의 매핑에 따라 'Normal' 클래스의 인코딩된 값 확인
# 일반적으로 LabelEncoder는 알파벳 순서로 매핑하므로 'Cancer':0, 'Normal':1 일 가능성이 높음
# 하지만 확실하게 inverse_transform을 통해 확인하여 positive_class_label을 설정
if le.inverse_transform([0])[0] == 'Normal':
    normal_label_encoded_value = 0
    cancer_label_encoded_value = 1
else:
    normal_label_encoded_value = 1
    cancer_label_encoded_value = 0

positive_class_label = normal_label_encoded_value # LightGBM의 scale_pos_weight는 positive class에 대한 가중치
neg_count = np.sum(y_meta == cancer_label_encoded_value) # Cancer 클래스 개수
pos_count = np.sum(y_meta == normal_label_encoded_value) # Normal 클래스 개수

class_weight = neg_count / pos_count if pos_count > 0 else 1.0 

print(f"\n메타 학습 데이터 준비 완료. 샘플 수: {len(X_meta)}, 특징 수: {len(X_meta.columns)}")
print(f"메타 데이터 레이블 분포: Cancer {neg_count}개 / Normal {pos_count}개 (Normal 가중치: {class_weight:.2f})")

# --- 3. 메타 모델 교차 검증 및 상세 분석 (하이퍼파라미터 튜닝 추가) ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_proba = np.zeros(len(X_meta)) # OOF 예측 확률 저장 (0~1)
val_scores = []
all_shap_values_folds = [] 
all_X_val_dfs = [] 

print("\n최종 메타 모델을 5-겹 교차 검증으로 학습 및 평가합니다...")

# 메타 모델의 하이퍼파라미터 튜닝을 위한 범위 설정
param_grid = {
    'n_estimators': [100, 200, 300, 500], # 학습 라운드 증가
    'learning_rate': [0.01, 0.02, 0.03, 0.05], # 학습률
    'num_leaves': [3, 5, 7, 10], # 트리 복잡도 증가
    'max_depth': [2, 3, 5], # 트리 깊이 증가
    'reg_alpha': [0, 0.01, 0.1, 0.5], # L1 정규화 (0부터 시작)
    'reg_lambda': [0, 0.01, 0.1, 0.5], # L2 정규화 (0부터 시작)
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0], # 특징 샘플링 비율
    'subsample': [0.7, 0.8, 0.9, 1.0], # 데이터 샘플링 비율
    'min_child_samples': [10, 20, 30], # 리프 노드 최소 샘플 수 (더 낮춰서 복잡도 증가 시도)
    'scale_pos_weight': [class_weight, 1.0, 0.5, 2.0] # 계산된 가중치 외에 다른 값도 튜닝
}

# F1-score (weighted)를 최적화 기준으로 사용
scorer = make_scorer(f1_score, average='weighted')

# RandomizedSearchCV 설정
# n_jobs=-1로 모든 코어 사용, verbose를 1로 설정하여 진행 상황 확인 (너무 자세한 것은 아님)
meta_random_search = RandomizedSearchCV(
    estimator=lgb.LGBMClassifier(objective='binary', random_state=42, verbose=-1),
    param_distributions=param_grid,
    n_iter=100, # 탐색할 조합의 수 (컴퓨팅 자원 고려)
    scoring=scorer,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    verbose=1, # 튜닝 진행 상황 출력
    n_jobs=-1,
    random_state=42
)

# 메타 모델 학습 데이터 전체로 RandomizedSearchCV 실행
print("\n메타 모델 하이퍼파라미터 튜닝 시작 (RandomizedSearchCV)...")
meta_random_search.fit(X_meta, y_meta)

best_params = meta_random_search.best_params_
print(f"\n최적의 메타 모델 하이퍼파라미터: {best_params}")
print(f"최고 F1 스코어 (RandomizedSearchCV): {meta_random_search.best_score_:.4f}")

# 최적의 하이퍼파라미터로 교차 검증 수행
for fold, (train_idx, val_idx) in tqdm(enumerate(cv.split(X_meta, y_meta)), total=5, desc="교차 검증 진행률"):
    print(f"\n--- 메타 모델 Fold {fold+1} 학습 및 예측 시작 (최적 파라미터 적용) ---")
    X_train, X_val = X_meta.iloc[train_idx], X_meta.iloc[val_idx]
    y_train, y_val = y_meta[train_idx], y_meta[val_idx]
    
    # 훈련 데이터를 다시 훈련/검증 세트로 분할하여 early stopping에 활용
    # RandomizedSearchCV 내부에서 이미 최적 파라미터를 찾았으므로, 이 단계는 교차 검증을 위한 일반적인 분할
    X_sub_train, X_sub_val, y_sub_train, y_sub_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    meta_model = lgb.LGBMClassifier(
        objective='binary', 
        random_state=42, 
        verbose=-1, 
        **best_params # 최적 하이퍼파라미터 적용
    )
    
    meta_model.fit(X_sub_train, y_sub_train,
                   eval_set=[(X_sub_val, y_sub_val)],
                   eval_metric='logloss', 
                   callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]) # stopping_rounds를 100으로 증가

    # 학습 곡선 시각화 함수 호출
    plot_learning_curve_custom(meta_model.evals_result_, fold, OUTPUT_DIR, MODEL_NAME_PREFIX, MODEL_TO_RUN)

    # 상세 성능 지표 출력
    train_pred_proba = meta_model.predict_proba(X_train)[:, 1] 
    print(f"   [Train] Acc: {accuracy_score(y_train, train_pred_proba > 0.5):.4f}, F1: {f1_score(y_train, train_pred_proba > 0.5):.4f}, AUC: {roc_auc_score(y_train, train_pred_proba):.4f}")

    val_pred_proba = meta_model.predict_proba(X_val)[:, 1]
    print(f"   [Valid] Acc: {accuracy_score(y_val, val_pred_proba > 0.5):.4f}, F1: {f1_score(y_val, val_pred_proba > 0.5):.4f}, AUC: {roc_auc_score(y_val, val_pred_proba):.4f}")
    
    oof_preds_proba[val_idx] = val_pred_proba 
    val_scores.append({
        'acc': accuracy_score(y_val, val_pred_proba > 0.5), 
        'f1': f1_score(y_val, val_pred_proba > 0.5), 
        'auc': roc_auc_score(y_val, val_pred_proba)
    })
    
    # SHAP 분석 (이진 분류 모델)
    print("    - SHAP 값 계산 중...")
    explainer = shap.TreeExplainer(meta_model)
    raw_shap_values = explainer.shap_values(X_val)

    if isinstance(raw_shap_values, list) and len(raw_shap_values) == 2:
        shap_values_to_store = raw_shap_values[normal_label_encoded_value]
        print(f"DEBUG SHAP (Fold {fold+1}): Retrieved SHAP values for class '{le.inverse_transform([normal_label_encoded_value])[0]}'. Shape: {shap_values_to_store.shape}")
    elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 2:
        shap_values_to_store = raw_shap_values
        print(f"DEBUG SHAP (Fold {fold+1}): Retrieved single 2D SHAP values directly. Shape: {shap_values_to_store.shape}")
    else:
        print(f"!!!!! WARNING SHAP (Fold {fold+1}): Unexpected SHAP values format: {type(raw_shap_values)}, Shape: {getattr(raw_shap_values, 'shape', 'N/A')}. Storing as is.")
        shap_values_to_store = raw_shap_values 

    all_shap_values_folds.append(shap_values_to_store)
    all_X_val_dfs.append(X_val) 
    joblib.dump(meta_model, os.path.join(OUTPUT_DIR, f'{MODEL_NAME_PREFIX}_{MODEL_TO_RUN}_model_fold_{fold+1}.pkl'))

# --- 4. 최종 결과 종합 및 시각화 ---
print("\n=============================================")
print("===   '암 vs 정상' 앙상블 모델 최종 성능    ===")
print("=============================================")
print(f"평균 검증(Validation) Accuracy: {np.mean([s['acc'] for s in val_scores]):.4f}")
print(f"평균 검증(Validation) F1-Score: {np.mean([s['f1'] for s in val_scores]):.4f}")
print(f"평균 검증(Validation) AUC: {np.mean([s['auc'] for s in val_scores]):.4f}")

oof_pred_labels = (oof_preds_proba > 0.5).astype(int)
target_names_ordered = [le.inverse_transform([0])[0], le.inverse_transform([1])[0]] 
print("\n[최종 분류 리포트]")
print(classification_report(y_meta, oof_pred_labels, target_names=target_names_ordered))

cm = confusion_matrix(y_meta, oof_pred_labels)
cm_df = pd.DataFrame(cm, index=target_names_ordered, columns=target_names_ordered) 
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('최종 앙상블 모델 혼동 행렬 (암 vs 정상)')
plt.savefig(os.path.join(OUTPUT_DIR, 'final_binary_ensemble_confusion_matrix.png'), bbox_inches='tight')
plt.close()
print("최종 혼동 행렬 그래프가 바탕화면에 저장되었습니다.")

# --- SHAP 기반 특징 중요도 분석 ---
print("\n--- 메타 모델 SHAP 기반 특징 중요도 분석 ---")
try:
    X_val_all_combined = pd.concat(all_X_val_dfs, ignore_index=False) 
    
    if all_shap_values_folds and isinstance(all_shap_values_folds[0], np.ndarray) and all_shap_values_folds[0].ndim == 2:
        shap_values_combined_final = np.concatenate(all_shap_values_folds, axis=0)
        print(f"DEBUG SHAP Combined: 2D SHAP values integrated. Resulting shape: {shap_values_combined_final.shape}")
    else:
        print("ERROR: SHAP 값의 반환 형태가 예상과 다릅니다. SHAP 분석을 수행할 수 없습니다.")
        shap_values_combined_final = None 

    if shap_values_combined_final is not None and \
       shap_values_combined_final.shape[0] == X_val_all_combined.shape[0] and \
       shap_values_combined_final.shape[1] == X_val_all_combined.shape[1]:
        
        # 1. 종합 특징 중요도
        shap.summary_plot(shap_values_combined_final, X_val_all_combined, plot_type="bar", show=False, max_display=20)
        plt.title("[메타 모델] 종합 특징 중요도 (Top 20)")
        plt.savefig(os.path.join(OUTPUT_DIR, 'meta_model_1_overall_feature_importance.png'), bbox_inches='tight')
        plt.close()
        print("종합 특징 중요도 그래프 저장 완료.")

        # 2. 전문가(오믹스)별 중요도
        importance_scores = np.abs(shap_values_combined_final).mean(axis=0) 
        
        feature_importance_df = pd.DataFrame({'feature': X_val_all_combined.columns, 'importance': importance_scores})
        
        feature_importance_df['omics_type'] = feature_importance_df['feature'].apply(
            lambda x: x.replace('pred_', '').replace('_cancer_prob', '').replace('_normal_prob', '') if x.startswith('pred_') else x 
        ) 
        
        omics_importance = feature_importance_df.groupby('omics_type')['importance'].sum().sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        omics_importance.plot(kind='bar', color='skyblue')
        plt.title('[메타 모델] 전문가(오믹스)별 중요도 총합')
        plt.ylabel('Total Mean(|SHAP Value|)')
        plt.xticks(rotation=0)
        plt.savefig(os.path.join(OUTPUT_DIR, 'meta_model_2_omics_type_importance.png'), bbox_inches='tight')
        plt.close()
        print("전문가(오믹스)별 중요도 그래프 저장 완료.")

    else:
        print("SHAP 분석을 위한 데이터가 유효하지 않습니다. 그래프 생성 건너뜜.")

except Exception as e:
    print(f"SHAP 기반 분석 중 오류 발생: {e}")

# --- 5. 최종 모델 저장 ---
print("\n전체 데이터로 최종 메타 모델을 재학습하고 저장합니다...")
final_meta_model = lgb.LGBMClassifier(objective='binary', random_state=42, verbose=-1, n_estimators=N_ESTIMATORS, scale_pos_weight=class_weight)
final_meta_model.fit(X_meta, y_meta)
joblib.dump(final_meta_model, os.path.join(DESKTOP_PATH, 'final_meta_model_binary.pkl'))
print("성공! 'final_meta_model_binary.pkl'이 바탕화면에 저장되었습니다.")