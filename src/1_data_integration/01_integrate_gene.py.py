# 1_integrate_gene_data_final.py
import pandas as pd
import os

print("--- [1단계] Gene 전문가 모델용 데이터 통합 시작 (v2. 최종 포맷 대응) ---")

# --- 설정 ---
DESKTOP_PATH = r"C:\Users\401-1\Desktop"

# 각 암종별 폴더와 파일 정보
# STAD는 이제 포함하지만, 내부 로직에서 fpkm_unstranded 값을 사용하도록 처리
CANCER_INFO = {
    'OV':   {'folder': 'TCGA-OV',   'file': 'merged_gene_expression_cancer.csv', 'gene_col': 'gene'},
    'BRCA': {'folder': 'TCGA-BRCA', 'file': 'merged_gene_expression_cancer.csv', 'gene_col': 'gene_id'},
    'STAD': {'folder': 'TCGA-STAD', 'file': 'merged_gene_expression_cancer.csv', 'gene_col': 'gene_id'},
    'KIRC': {'folder': 'TCGA-KIRC', 'file': 'merged_gene_expression_cancer.csv', 'gene_col': 'gene'},
    'LUSC': {'folder': 'TCGA-LUSC', 'file': 'merged_gene_expression_cancer.csv', 'gene_col': 'gene'},
    'LIHC': {'folder': 'TCGA-LIHC', 'file': 'merged_gene_expression_cancer.csv', 'gene_col': 'gene'}
}

OUTPUT_FILENAME = "gene_multiclass_basemodel_data.csv"
# -----------

# 모든 암종의 데이터를 표준화된 'Long' 포맷으로 담을 리스트
long_format_dfs = []
# 각 샘플이 어떤 암종인지 매핑하는 딕셔너리
sample_to_cancer_map = {}

print("각 암종별 파일 처리 시작...")
for cancer_code, info in CANCER_INFO.items():
    file_path = os.path.join(DESKTOP_PATH, info['folder'], info['file'])
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"-> '{cancer_code}' 데이터 처리 중...")
        
        # 1. 필수 컬럼만 선택 (환자ID, 유전자ID, 발현값)
        #    파일마다 다른 유전자 ID 컬럼 이름을 info['gene_col']로 통일
        required_cols = ['patient_barcode', info['gene_col'], 'fpkm_unstranded']
        df_subset = df[required_cols].copy()
        
        # 2. 컬럼 이름 표준화
        df_subset.columns = ['patient_barcode', 'gene_id', 'expression_value']
        
        # 3. 유효한 유전자 데이터만 필터링 (ENSG로 시작하는 것만 남김)
        df_subset.dropna(subset=['gene_id'], inplace=True) # gene_id가 없는 행 제거
        df_subset = df_subset[df_subset['gene_id'].str.startswith('ENSG')]
        
        # 4. 숫자형 데이터 변환 및 결측치 처리
        df_subset['expression_value'] = pd.to_numeric(df_subset['expression_value'], errors='coerce').fillna(0)
        
        # 5. 샘플-암종 매핑 정보 저장 및 리스트에 추가
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
    # 6. 모든 암종의 Long 포맷 데이터를 하나로 합치기
    print("\n모든 암종의 데이터를 하나로 통합 중...")
    long_final_df = pd.concat(long_format_dfs, ignore_index=True)
    
    # 7. Pivot을 사용하여 Long 포맷을 Wide 포맷으로 변환 (샘플 x 특징)
    print("Pivot을 사용하여 데이터를 (샘플 x 특징) 형태로 변환 중... (시간이 걸릴 수 있습니다)")
    wide_df = long_final_df.pivot_table(
        index='patient_barcode', 
        columns='gene_id', 
        values='expression_value',
        fill_value=0 # Pivot 후 생기는 결측치는 0으로 채움
    )
    
    # 8. 최종 Wide 포맷 데이터에 'label' 컬럼 추가
    wide_df['label'] = wide_df.index.map(sample_to_cancer_map)
    
    # 결과 확인
    print("\n--- 통합 결과 ---")
    # 마지막 컬럼이 label이므로 -1
    print(f"통합 후 데이터 크기: {wide_df.shape[0]}개의 샘플, {wide_df.shape[1]-1}개의 유전자 특징")
    print("\n암종별 샘플 수:")
    print(wide_df['label'].value_counts())

    # 최종 파일 저장
    output_path = os.path.join(DESKTOP_PATH, OUTPUT_FILENAME)
    wide_df.to_csv(output_path)
    print(f"\n성공! 통합된 파일이 바탕화면에 '{OUTPUT_FILENAME}' 이름으로 저장되었습니다.")