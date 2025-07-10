# 1_integrate_meth_data.py (v2: STAD 포함)
import pandas as pd
import os

print("--- [1단계] Meth 전문가 모델용 데이터 통합 시작 (STAD 포함) ---")

# --- 설정 ---
DESKTOP_PATH = r"C:\Users\401-1\Desktop"

# 통합할 암종과 파일 정보 (OV 제외, STAD 포함)
CANCER_FILES = {
    'BRCA': os.path.join(DESKTOP_PATH, 'TCGA-BRCA', 'merged_breast_methylation_cancer.csv'),
    'KIRC': os.path.join(DESKTOP_PATH, 'TCGA-KIRC', 'merged_kidney_methylation_cancer.csv'),
    'LUSC': os.path.join(DESKTOP_PATH, 'TCGA-LUSC', 'merged_lusc_methylation_cancer.csv'),
    'LIHC': os.path.join(DESKTOP_PATH, 'TCGA-LIHC', 'merged_methylation_cancer.csv'),
    'STAD': os.path.join(DESKTOP_PATH, 'TCGA-STAD', 'merged_methylation_cancer.csv') # <-- STAD 추가
}

OUTPUT_FILENAME = "meth_multiclass_basemodel_data.csv"
# -----------

all_dfs = []
print("각 암종별 파일 처리 시작...")
for cancer_code, file_path in CANCER_FILES.items():
    try:
        print(f"-> '{cancer_code}' 데이터 처리 중...")
        # STAD 파일의 컬럼 수가 다를 수 있으므로 low_memory=False 옵션 추가
        df = pd.read_csv(file_path, low_memory=False) 
        
        # 불필요한 컬럼 제거
        cols_to_drop = [col for col in df.columns if col not in ['patient_barcode'] and not col.startswith('cg')]
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        df['label'] = cancer_code
        all_dfs.append(df)
    except FileNotFoundError:
        print(f"!!! 경고: '{file_path}' 파일을 찾을 수 없습니다. 건너뜁니다.")

if not all_dfs:
    print("\n오류: 처리할 데이터가 없습니다. 작업을 중단합니다.")
else:
    # patient_barcode를 인덱스로 설정
    for i, df in enumerate(all_dfs):
        all_dfs[i] = df.set_index('patient_barcode')

    # 모든 데이터프레임을 하나로 통합 (공통된 cg 프로브만 사용)
    print("\n모든 데이터프레임을 하나로 통합합니다 (join='inner')...")
    final_meth_df = pd.concat(all_dfs, join='inner')
    
    # 결과 확인
    print("\n--- 통합 결과 ---")
    print(f"통합 후 데이터 크기: {final_meth_df.shape[0]}개의 샘플, {final_meth_df.shape[1]-1}개의 메틸레이션 특징")
    print("\n암종별 샘플 수:")
    print(final_meth_df['label'].value_counts())

    # 최종 파일 저장
    output_path = os.path.join(DESKTOP_PATH, OUTPUT_FILENAME)
    final_meth_df.to_csv(output_path)
    print(f"\n성공! 통합된 파일이 바탕화면에 '{OUTPUT_FILENAME}' 이름으로 저장되었습니다.")