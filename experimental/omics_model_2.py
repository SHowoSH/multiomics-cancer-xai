# 모델 학습 및 저장용 예시 코드
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import json
import os

# 1. 학습 데이터 로드
X = pd.read_csv(r"C:\Users\401-1\Desktop\final_sample_based_features_omics_only_revised.csv", index_col='sample_id')
y = pd.read_csv(r"C:\Users\401-1\Desktop\final_sample_based_target_omics_only_revised.csv", index_col='sample_id')['target']

# 2. 모델 학습
final_model = RandomForestClassifier(random_state=42)
final_model.fit(X, y)
print("모델 학습 완료!")

# 3. 최종 산출물 3종 세트 저장
FINAL_ARTIFACTS_DIR = "final_model_artifacts"
os.makedirs(FINAL_ARTIFACTS_DIR, exist_ok=True)

# 3-1. 최종 모델 저장
joblib.dump(final_model, os.path.join(FINAL_ARTIFACTS_DIR, "final_model.joblib"))

# 3-2. 최종 Imputer 저장 (최종 X 데이터로 fit)
final_imputer = SimpleImputer(strategy='median').fit(X)
joblib.dump(final_imputer, os.path.join(FINAL_ARTIFACTS_DIR, "imputer.joblib"))

# 3-3. 최종 특징 이름 목록 저장
with open(os.path.join(FINAL_ARTIFACTS_DIR, "feature_names.json"), 'w') as f:
    json.dump(list(X.columns), f)
    
print(f"'{FINAL_ARTIFACTS_DIR}' 폴더에 최종 모델 및 관련 파일 저장 완료.")