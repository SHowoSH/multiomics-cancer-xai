# preprocessing_utils.py
import pandas as pd

def load_feature_list(path):
    """특징 목록 파일을 읽어 리스트로 반환하는 함수"""
    with open(path, 'r') as f:
        features = [line.strip() for line in f]
    return features

def preprocess_long_format(raw_df, feature_list, id_col, feature_col, value_col):
    """Long 포맷 데이터를 모델 입력용 Wide 포맷으로 변환"""
    df_subset = raw_df[[id_col, feature_col, value_col]].copy()
    df_subset.columns = ['patient_id', 'feature', 'value']
    df_subset['value'] = pd.to_numeric(df_subset['value'], errors='coerce').fillna(0)
    
    wide_df = df_subset.pivot_table(
        index='patient_id', 
        columns='feature', 
        values='value',
        fill_value=0
    )
    
    # 저장된 특징 목록과 순서를 정확히 맞춤
    processed_df = wide_df.reindex(columns=feature_list, fill_value=0)
    return processed_df

def preprocess_mutation_data(raw_df, feature_list):
    """Mutation 데이터를 바이너리 Wide 포맷으로 변환"""
    df_subset = raw_df[['patient_barcode', 'Hugo_Symbol']].copy()
    df_subset.drop_duplicates(inplace=True)
    df_subset['value'] = 1
    
    wide_df = df_subset.pivot_table(
        index='patient_barcode', 
        columns='Hugo_Symbol', 
        values='value',
        fill_value=0
    )
    
    processed_df = wide_df.reindex(columns=feature_list, fill_value=0)
    return processed_df

def preprocess_methylation_data(raw_df, common_feature_list):
    """Wide 포맷의 메틸레이션 데이터를 정리"""
    if 'patient_barcode' not in raw_df.columns:
        return None

    df = raw_df.set_index('patient_barcode')
    
    cols_to_drop = [col for col in df.columns if not col.startswith('cg')]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    processed_df = df.reindex(columns=common_feature_list, fill_value=0)
    return processed_df