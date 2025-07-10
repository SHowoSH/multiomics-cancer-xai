# 1_integrate_mut_data.py
import pandas as pd
import os

print("--- [1단계] Mutation 전문가 모델용 데이터 통합 및 변환 시작 ---")

# --- 설정 ---
DESKTOP_PATH = r"C:\Users\401-1\Desktop"

# 통합할 암종과 파일 정보 (6개 암종 전체)
CANCER_INFO = {
    'OV':   {'folder': 'TCGA-OV',   'file': 'merged_ov_mutation_cancer.csv'},
    'BRCA': {'folder': 'TCGA-BRCA', 'file': 'merged_breast_mutation_cancer.csv'},
    'STAD': {'folder': 'TCGA-STAD', 'file': 'merged_stomach_mutation_cancer.csv'},
    'KIRC': {'folder': 'TCGA-KIRC', 'file': 'merged_kidney_mutation_cancer.csv'},
    'LUSC': {'folder': 'TCGA-LUSC', 'file': 'merged_lusc_mutation_cancer.csv'},
    'LIHC': {'folder': 'TCGA-LIHC', 'file': 'merged_mutation_cancer.csv'}
}

OUTPUT_FILENAME = "mutation_multiclass_basemodel_data.csv"
# -----------

long_format_dfs = []
sample_to_cancer_map = {}

print("각 암종별 파일 처리 시작...")
for cancer_code, info in CANCER_INFO.items():
    file_path = os.path.join(DESKTOP_PATH, info['folder'], info['file'])
    try:
        print(f"-> '{cancer_code}' 데이터 처리 중...")
        df = pd.read_csv(file_path, low_memory=False)
        
        # 1. 필수 컬럼만 선택: 'patient_barcode', 'Hugo_Symbol'
        df_subset = df[['patient_barcode', 'Hugo_Symbol']].copy()
        
        # 2. 중복 제거: 한 환자에게 같은 유전자 변이가 여러번 기록된 경우, 하나만 남김
        df_subset.drop_duplicates(inplace=True)
        
        # 3. 샘플-암종 매핑 정보 저장 및 리스트에 추가
        for sample in df_subset['patient_barcode'].unique():
            sample_to_cancer_map[sample] = cancer_code
        long_format_dfs.append(df_subset)
        
    except FileNotFoundError:
        print(f"!!! 경고: '{file_path}' 파일을 찾을 수 없습니다. 건너뜁니다.")
    except KeyError as e:
        print(f"!!! 오류: '{file_path}' 파일에 필요한 컬럼({e})이 없습니다. 건너뜁니다.")

if not long_format_dfs:
    print("\n오류: 처리할 데이터가 없습니다. 작업을 중단합니다.")
else:
    # 4. 모든 암종 데이터를 하나로 통합하고, 변이가 존재함을 '1'로 표시
    print("\n모든 암종의 데이터를 하나로 통합 중...")
    long_final_df = pd.concat(long_format_dfs, ignore_index=True)
    long_final_df['value'] = 1 # 변이가 있으면 1
    
    # 5. Pivot을 사용하여 Long 포맷을 Wide 포맷으로 변환 (바이너리 행렬 생성)
    print("Pivot을 사용하여 (샘플 x 특징) 형태로 변환 중...")
    wide_df = long_final_df.pivot_table(
        index='patient_barcode', 
        columns='Hugo_Symbol', 
        values='value',
        fill_value=0 # 변이가 없는 칸은 0으로 채움
    )
    
    # 6. 'label' 컬럼 추가
    wide_df['label'] = wide_df.index.map(sample_to_cancer_map)
    
    # 결과 확인
    print("\n--- 통합 결과 ---")
    print(f"통합 후 데이터 크기: {wide_df.shape[0]}개의 샘플, {wide_df.shape[1]-1}개의 유전자 특징")
    print("\n암종별 샘플 수:")
    print(wide_df['label'].value_counts())

    # 최종 파일 저장
    output_path = os.path.join(DESKTOP_PATH, OUTPUT_FILENAME)
    wide_df.to_csv(output_path)
    print(f"\n성공! 통합된 파일이 바탕화면에 '{OUTPUT_FILENAME}' 이름으로 저장되었습니다.")