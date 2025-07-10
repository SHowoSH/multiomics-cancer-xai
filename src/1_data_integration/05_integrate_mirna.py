# 1_integrate_mirna_data.py
import pandas as pd
import os

print("--- [1단계] miRNA 전문가 모델용 데이터 통합 및 변환 시작 ---")

# --- 설정 ---
DESKTOP_PATH = r"C:\Users\401-1\Desktop"

# 포맷별로 파일 정보 분리
LONG_FORMAT_CANCERS = {
    'OV':   {'folder': 'TCGA-OV',   'file': 'merged_mirna_expression_cancer.csv'},
    'BRCA': {'folder': 'TCGA-BRCA', 'file': 'merged_breast_mirna_cancer.csv'},
    'KIRC': {'folder': 'TCGA-KIRC', 'file': 'merged_kidney_mirna_expression_cancer.csv'},
    'LUSC': {'folder': 'TCGA-LUSC', 'file': 'merged_mirna_expression_cancer.csv'},
    'LIHC': {'folder': 'TCGA-LIHC', 'file': 'merged_mirna_expression_cancer.csv'}
}
WIDE_FORMAT_CANCERS = {
    'STAD': {'folder': 'TCGA-STAD', 'file': 'merged_mirna_cancer.csv'}
}
OUTPUT_FILENAME = "mirna_multiclass_basemodel_data.csv"
# -----------

# --- 1. Long 포맷 데이터 처리 ---
long_format_dfs = []
sample_to_cancer_map = {}

print("Long 포맷 파일 처리 시작 (5개 암종)...")
for cancer_code, info in LONG_FORMAT_CANCERS.items():
    file_path = os.path.join(DESKTOP_PATH, info['folder'], info['file'])
    try:
        df = pd.read_csv(file_path, low_memory=False)
        # 필수 컬럼 선택 및 표준화
        df_subset = df[['patient_barcode', 'miRNA_ID', 'reads_per_million_miRNA_mapped']].copy()
        df_subset.columns = ['patient_barcode', 'miRNA_ID', 'value']
        # 샘플-암종 매핑 및 리스트 추가
        for sample in df_subset['patient_barcode'].unique():
            sample_to_cancer_map[sample] = cancer_code
        long_format_dfs.append(df_subset)
    except Exception as e:
        print(f"!!! '{file_path}' 처리 중 오류: {e}. 건너뜁니다.")

# Long 포맷 데이터 통합 및 Wide 포맷으로 변환
if long_format_dfs:
    long_combined_df = pd.concat(long_format_dfs, ignore_index=True)
    print("Pivot을 사용하여 Long 포맷 -> Wide 포맷 변환 중...")
    wide_df_others = long_combined_df.pivot_table(
        index='patient_barcode', columns='miRNA_ID', values='value', fill_value=0
    )
    wide_df_others['label'] = wide_df_others.index.map(sample_to_cancer_map)
else:
    wide_df_others = pd.DataFrame()


# --- 2. Wide 포맷 데이터 처리 ---
print("\nWide 포맷 파일 처리 시작 (STAD)...")
try:
    stad_info = WIDE_FORMAT_CANCERS['STAD']
    stad_file_path = os.path.join(DESKTOP_PATH, stad_info['folder'], stad_info['file'])
    stad_df_wide = pd.read_csv(stad_file_path)
    stad_df_wide.set_index('patient_barcode', inplace=True)
    stad_df_wide['label'] = 'STAD'
except Exception as e:
    print(f"!!! '{stad_file_path}' 처리 중 오류: {e}.")
    stad_df_wide = pd.DataFrame()


# --- 3. 모든 Wide 포맷 데이터 통합 ---
if not wide_df_others.empty and not stad_df_wide.empty:
    print("\n최종 데이터 통합 (5개 암종 + STAD)...")
    # 공통된 miRNA 특징만 남기기 위해 inner join 사용
    final_mirna_df = pd.concat([wide_df_others, stad_df_wide], join='inner')
    
    # 결과 확인
    print("\n--- 통합 결과 ---")
    print(f"통합 후 데이터 크기: {final_mirna_df.shape[0]}개의 샘플, {final_mirna_df.shape[1]-1}개의 miRNA 특징")
    print("\n암종별 샘플 수:")
    print(final_mirna_df['label'].value_counts())

    # 최종 파일 저장
    output_path = os.path.join(DESKTOP_PATH, OUTPUT_FILENAME)
    final_mirna_df.to_csv(output_path)
    print(f"\n성공! 통합된 파일이 바탕화면에 '{OUTPUT_FILENAME}' 이름으로 저장되었습니다.")
else:
    print("\n오류: 데이터가 부족하여 최종 통합을 수행할 수 없습니다.")