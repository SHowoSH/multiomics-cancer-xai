import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import re
import joblib
import os

# --- 0. 전역 상수 및 설정 ---
PATIENT_ID_COL = 'bcr_patient_barcode'
PCA_MODELS_DIR = "pca_models_and_features"

if not os.path.exists(PCA_MODELS_DIR):
    os.makedirs(PCA_MODELS_DIR)
    print(f"디렉토리 생성됨: {PCA_MODELS_DIR}")

# --- 1. 임상 데이터 처리 함수들 (이 부분은 더 이상 사용되지 않지만, 함수 정의는 유지) ---
RELEVANT_CLINICAL_COLS = [
    PATIENT_ID_COL, 'age_at_initial_pathologic_diagnosis', 'gender', 'ethnicity', 'race_list_race',
    'menopause_status', 'histological_type', 'stage_event_pathologic_stage',
    'stage_event_tnm_categories_pathologic_categories_pathologic_T',
    'stage_event_tnm_categories_pathologic_categories_pathologic_N',
    'stage_event_tnm_categories_pathologic_categories_pathologic_M',
    'breast_carcinoma_estrogen_receptor_status', 'breast_carcinoma_progesterone_receptor_status',
    'lab_proc_her2_neu_immunohistochemistry_receptor_status', 'number_of_lymphnodes_positive_by_he',
    'margin_status', 'history_of_neoadjuvant_treatment'
]

def load_clinical_data(file_path):
    print("경고: 임상 데이터 처리가 비활성화되어, load_clinical_data 함수는 호출되지 않습니다.")
    return pd.DataFrame() # 빈 데이터프레임 반환

def select_and_clean_clinical_data(df_full, selected_columns):
    print("경고: 임상 데이터 처리가 비활성화되어, select_and_clean_clinical_data 함수는 호출되지 않습니다.")
    return pd.DataFrame() # 빈 데이터프레임 반환

def preprocess_clinical_features_for_merging(df_clinical_selected):
    print("경고: 임상 데이터 처리가 비활성화되어, preprocess_clinical_features_for_merging 함수는 호출되지 않습니다.")
    return pd.DataFrame() # 빈 데이터프레임 반환


# --- 오믹스 데이터 처리 함수 (이전과 동일, PCA 로직 없음) ---
def process_single_omics_file(file_path, patient_id_col_name, id_col_for_pivot, value_col,
                              omics_name, log_transform=False, filter_low_variance_quantile=0.05,
                              apply_scaling=True, aggfunc='mean',
                              omics_file_type='expression',
                              variant_class_col=None, min_mutation_freq_percent=1.0):
    print(f"  > {omics_name} ({omics_file_type}) 파일 기본 처리 시작: {file_path}")
    if not file_path or not os.path.exists(file_path):
        print(f"    경고: 파일 없음 또는 경로 오류 - {file_path}"); return pd.DataFrame()
    df_processed = pd.DataFrame()
    try:
        df_raw = pd.read_csv(file_path, low_memory=False)
        original_patient_ids_in_file = None
        if patient_id_col_name in df_raw.columns:
            df_raw.rename(columns={patient_id_col_name: PATIENT_ID_COL}, inplace=True)
            df_raw[PATIENT_ID_COL] = df_raw[PATIENT_ID_COL].astype(str).str.upper() # 대문자 통일
            original_patient_ids_in_file = df_raw[PATIENT_ID_COL].unique()
            df_raw.set_index(PATIENT_ID_COL, inplace=True)
        elif PATIENT_ID_COL == df_raw.index.name:
            df_raw.index = df_raw.index.astype(str).str.upper()
            original_patient_ids_in_file = df_raw.index.unique()
        else:
            print(f"    경고: {PATIENT_ID_COL} 컬럼이 {file_path}에 명시적으로 없습니다. 파일 형식을 확인하세요."); return pd.DataFrame()

        if omics_file_type == 'mutation':
            df_raw_reset = df_raw.reset_index()
            required_cols_in_df = [PATIENT_ID_COL, id_col_for_pivot, variant_class_col]
            if not all(col in df_raw_reset.columns for col in required_cols_in_df):
                missing = [col for col in required_cols_in_df if col not in df_raw_reset.columns]
                print(f"    오류: 변이 데이터 필요 컬럼 부족 ({', '.join(missing)})."); return pd.DataFrame(index=original_patient_ids_in_file)

            meaningful_variants = ['Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Missense_Mutation', 'Nonsense_Mutation', 'Splice_Site', 'Translation_Start_Site', 'Nonstop_Mutation']
            df_filtered = df_raw_reset[df_raw_reset[variant_class_col].isin(meaningful_variants)][required_cols_in_df].copy()
            df_filtered.dropna(subset=[PATIENT_ID_COL, id_col_for_pivot], inplace=True)
            df_filtered[PATIENT_ID_COL] = df_filtered[PATIENT_ID_COL].astype(str)
            df_filtered[id_col_for_pivot] = df_filtered[id_col_for_pivot].astype(str)
            df_filtered = df_filtered.drop_duplicates(subset=[PATIENT_ID_COL, id_col_for_pivot])

            if df_filtered.empty: print(f"    {file_path}: 필터링 후 변이 데이터 없음"); return pd.DataFrame(index=original_patient_ids_in_file)
            df_wide = pd.crosstab(df_filtered[PATIENT_ID_COL], df_filtered[id_col_for_pivot])
            df_wide = (df_wide > 0).astype(int)
            if df_wide.empty: return pd.DataFrame(index=df_filtered[PATIENT_ID_COL].unique())

            if min_mutation_freq_percent > 0 and df_wide.shape[0] > 0:
                min_count = int(np.ceil(len(df_wide) * (min_mutation_freq_percent / 100.0))); min_count = max(1, min_count)
                df_wide = df_wide.loc[:, df_wide.sum(axis=0) >= min_count]
            df_processed = df_wide

        elif omics_file_type == 'methylation':
            df_wide = df_raw.select_dtypes(include=np.number)
            if df_wide.empty: print(f"    {file_path}: 수치형 메틸레이션 데이터 없음"); return pd.DataFrame(index=original_patient_ids_in_file)
            df_wide = df_wide.apply(lambda x: x.fillna(x.median()), axis=0)
            df_processed = df_wide

        elif omics_file_type in ['expression', 'mirna', 'cnv']:
            df_raw_reset = df_raw.reset_index()
            required_cols_in_df = [PATIENT_ID_COL, id_col_for_pivot, value_col]
            if not all(col in df_raw_reset.columns for col in required_cols_in_df):
                missing = [col for col in required_cols_in_df if col not in df_raw_reset.columns]
                print(f"    오류: {omics_name} 데이터 필요 컬럼 부족 ({', '.join(missing)})."); return pd.DataFrame(index=original_patient_ids_in_file)

            df_selected = df_raw_reset[required_cols_in_df].copy()
            df_selected.dropna(subset=[value_col, id_col_for_pivot], inplace=True)
            df_selected[id_col_for_pivot] = df_selected[id_col_for_pivot].astype(str)

            df_wide = df_selected.pivot_table(index=PATIENT_ID_COL, columns=id_col_for_pivot, values=value_col, aggfunc=aggfunc)
            df_wide.fillna(0, inplace=True)
            df_processed = df_wide
        else:
            print(f"    오류: 알 수 없는 오믹스 파일 타입 - {omics_file_type}"); return pd.DataFrame(index=original_patient_ids_in_file)

        if df_processed.empty:
            print(f"    {omics_name} 초기 처리 후 데이터 없음."); return pd.DataFrame(index=original_patient_ids_in_file)

        print(f"    초기 전처리 후 형상 ({omics_name}): {df_processed.shape}")

        if omics_file_type != 'mutation':
            if log_transform: df_processed = np.log2(df_processed + 1)
            if filter_low_variance_quantile is not None and 0 < filter_low_variance_quantile < 1 and df_processed.shape[1] > 1:
                variances = df_processed.var(axis=0)
                if not variances.empty and variances.nunique() > 1:
                    variance_threshold = variances.quantile(filter_low_variance_quantile)
                    df_processed = df_processed.loc[:, variances > variance_threshold]
                    print(f"    분산 필터링 후 형상 ({omics_name}): {df_processed.shape}")
                elif not variances.empty:
                    df_processed = df_processed.loc[:, variances > 0]
                    print(f"    분산 0인 특징 제거 후 형상 ({omics_name}): {df_processed.shape}")

            if df_processed.empty or df_processed.shape[1] == 0:
                print(f"    {omics_name} 분산 필터링 후 데이터 없음."); return pd.DataFrame(index=df_processed.index)

            if apply_scaling:
                scaler = StandardScaler()
                df_processed_scaled_values = scaler.fit_transform(df_processed)
                scaler_filename = os.path.join(PCA_MODELS_DIR, f"{omics_name}_scaler.joblib")
                joblib.dump(scaler, scaler_filename)
                print(f"    스케일러 저장 완료: {scaler_filename}")
                df_processed = pd.DataFrame(df_processed_scaled_values, columns=df_processed.columns, index=df_processed.index)
        
        df_processed = df_processed.add_prefix(f"{omics_name}_")

    except FileNotFoundError: print(f"    경고: 파일 없음 - {file_path}"); return pd.DataFrame()
    except Exception as e:
        print(f"    오류 ({file_path} 처리 중): {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame()

    print(f"  > {omics_name} ({omics_file_type}) 파일 기본 처리 완료. 최종 형상: {df_processed.shape}")
    if original_patient_ids_in_file is not None:
        df_processed = df_processed.reindex(original_patient_ids_in_file.astype(str))

    return df_processed


# --- 3. 메인 전처리 실행 함수 (오믹스 데이터만 사용, PCA Imputer 저장 로직 추가) ---
def main_preprocessing_revised_omics_only():
    print("===== 샘플 기반 '암 vs 정상' 전처리 파이프라인 시작 (오믹스 데이터만 사용, 특징 유출 방지 버전) =====")

    omics_files_info = {
        'GeneExp': {'normal': r"", 'cancer': r"", 'id_col': 'gene_name', 'value_col': 'tpm_unstranded', 'log': True, 'pca': 100, 'type': 'expression', 'var_filter': 0.05, 'scale': True},
        'miRNA': {'normal': r"", 'cancer': r"", 'id_col': 'miRNA_ID', 'value_col': 'reads_per_million_miRNA_mapped', 'log': True, 'pca': 50, 'type': 'mirna', 'var_filter': 0.05, 'scale': True},
        'CNV': {'normal': r"C:\Users\401-1\Desktop\TCGA-STAD\merged_stomach_cnv_normal.csv", 'cancer': r"C:\Users\401-1\Desktop\TCGA-STAD\merged_stomach_cnv_cancer.csv", 'id_col': 'gene_name', 'value_col': 'copy_number', 'log': False, 'pca': 50, 'type': 'cnv', 'var_filter': 0.05, 'scale': True},
        'Meth': {'normal': r"", 'cancer': r"", 'id_col': None, 'value_col': None, 'log': False, 'pca': 100, 'type': 'methylation', 'var_filter': 0.05, 'scale': True},
        'Mut': {'normal': r"C:\Users\401-1\Desktop\TCGA-STAD\merged_stomach_mutation_normal.csv", 'cancer': r"C:\Users\401-1\Desktop\TCGA-STAD\merged_stomach_mutation_cancer.csv", 'id_col': 'Hugo_Symbol', 'value_col': None, 'variant_class_col': 'Variant_Classification', 'log': False, 'pca': None, 'type': 'mutation', 'var_filter': None, 'scale': False}
    }
    omics_pid_col_name = 'patient_barcode'

    X_clinical_processed_patient_indexed = pd.DataFrame()
    print("참고: 임상 데이터 처리는 이 버전에서 비활성화되었습니다.")

    # 2. 각 오믹스 데이터 파일별 기본 전처리 (PCA 이전)
    processed_omics_data_no_pca = {}
    for omics_name, info in omics_files_info.items():
        processed_omics_data_no_pca[omics_name] = {}
        for sample_category in ['normal', 'cancer']:
            file_path = info.get(sample_category)
            if file_path:
                df_temp = process_single_omics_file(
                    file_path=file_path, patient_id_col_name=omics_pid_col_name,
                    id_col_for_pivot=info['id_col'], value_col=info['value_col'],
                    omics_name=omics_name, log_transform=info['log'],
                    filter_low_variance_quantile=info.get('var_filter'),
                    apply_scaling=info.get('scale', True),
                    omics_file_type=info['type'], variant_class_col=info.get('variant_class_col'),
                    min_mutation_freq_percent=1.0 if info['type'] == 'mutation' else None
                )
                if df_temp is not None and not df_temp.empty:
                    processed_omics_data_no_pca[omics_name][sample_category] = df_temp
                elif df_temp is None:
                    print(f"경고: {omics_name} {sample_category} 처리 중 오류로 None 반환됨.")

    # 3. PCA 적용 (필요한 경우, normal/cancer 결합 데이터 사용)
    final_processed_omics_data = {}
    for omics_name, info in omics_files_info.items():
        final_processed_omics_data[omics_name] = {}
        df_normal_no_pca = processed_omics_data_no_pca.get(omics_name, {}).get('normal')
        df_cancer_no_pca = processed_omics_data_no_pca.get(omics_name, {}).get('cancer')
        n_pca_components = info.get('pca')

        apply_pca_condition = n_pca_components is not None and \
                              ( (df_normal_no_pca is not None and not df_normal_no_pca.empty) or \
                                (df_cancer_no_pca is not None and not df_cancer_no_pca.empty) )

        if apply_pca_condition:
            print(f"\n--- {omics_name}: PCA 적용 시작 (N_components={n_pca_components}) ---")
            dfs_to_concat_for_pca_fit = []
            if df_normal_no_pca is not None and not df_normal_no_pca.empty: dfs_to_concat_for_pca_fit.append(df_normal_no_pca)
            if df_cancer_no_pca is not None and not df_cancer_no_pca.empty: dfs_to_concat_for_pca_fit.append(df_cancer_no_pca)

            if not dfs_to_concat_for_pca_fit:
                print(f"    정보: {omics_name}에 PCA를 적용할 데이터가 없습니다 (dfs_to_concat_for_pca_fit 비어있음). 건너<0xEB>니다.")
                if df_normal_no_pca is not None: final_processed_omics_data[omics_name]['normal'] = df_normal_no_pca
                if df_cancer_no_pca is not None: final_processed_omics_data[omics_name]['cancer'] = df_cancer_no_pca
                continue
            
            all_cols = set()
            for df in dfs_to_concat_for_pca_fit:
                all_cols.update(df.columns)
            all_cols = sorted(list(all_cols))

            aligned_dfs_for_concat = []
            for df in dfs_to_concat_for_pca_fit:
                aligned_dfs_for_concat.append(df.reindex(columns=all_cols))
            
            df_combined_for_pca_fit = pd.concat(aligned_dfs_for_concat, axis=0)

            imputer_pca = SimpleImputer(strategy='median')
            num_cols_for_pca_imputation = df_combined_for_pca_fit.select_dtypes(include=np.number).columns
            if not num_cols_for_pca_imputation.empty:
                df_combined_for_pca_fit[num_cols_for_pca_imputation] = imputer_pca.fit_transform(df_combined_for_pca_fit[num_cols_for_pca_imputation])

                # --- **여기에 imputer_pca 저장 로직 추가** ---
                imputer_pca_filename = os.path.join(PCA_MODELS_DIR, f"{omics_name}_pca_imputer.joblib")
                joblib.dump(imputer_pca, imputer_pca_filename)
                print(f"    PCA용 Imputer 저장 완료: {imputer_pca_filename}")
                # --- 추가 끝 ---

            else:
                print(f"    경고: {omics_name} PCA용 결합 데이터에 숫자형 컬럼이 없어 imputation 생략.")


            if df_combined_for_pca_fit.empty or df_combined_for_pca_fit.shape[1] == 0:
                print(f"    경고: {omics_name} PCA 학습용 데이터(결합 및 imputation 후)가 비었거나 특징이 없습니다. 건너<0xEB>니다.")
                if df_normal_no_pca is not None: final_processed_omics_data[omics_name]['normal'] = df_normal_no_pca
                if df_cancer_no_pca is not None: final_processed_omics_data[omics_name]['cancer'] = df_cancer_no_pca
                continue

            current_n_samples, current_n_features = df_combined_for_pca_fit.shape
            max_possible_components = min(current_n_samples, current_n_features)
            adjusted_n_components = int(n_pca_components)
            if adjusted_n_components > max_possible_components:
                print(f"    경고: ({omics_name}) PCA 컴포넌트 수({adjusted_n_components}) > 최대({max_possible_components}), {max_possible_components}로 조정.")
                adjusted_n_components = max_possible_components
            
            if adjusted_n_components <= 0:
                print(f"    경고: ({omics_name}) 조정된 PCA 컴포넌트 수가 0 이하. PCA 건너<0xEB>니다.")
                if df_normal_no_pca is not None: final_processed_omics_data[omics_name]['normal'] = df_normal_no_pca
                if df_cancer_no_pca is not None: final_processed_omics_data[omics_name]['cancer'] = df_cancer_no_pca
                continue

            pca = PCA(n_components=adjusted_n_components)
            try:
                pca.fit(df_combined_for_pca_fit)
                print(f"    {omics_name}: PCA 모델 학습 완료. 설명된 분산 비율 합: {np.sum(pca.explained_variance_ratio_):.4f}")
                pca_model_filename = os.path.join(PCA_MODELS_DIR, f"{omics_name}_pca_model.joblib")
                original_features_filename = os.path.join(PCA_MODELS_DIR, f"{omics_name}_pre_pca_features.csv")
                joblib.dump(pca, pca_model_filename)
                pd.DataFrame({'feature': df_combined_for_pca_fit.columns.tolist()}).to_csv(original_features_filename, index=False)
                print(f"    PCA 모델 저장: {pca_model_filename}, 원본 특징명 저장: {original_features_filename}")

                for sample_cat, df_orig_no_pca in [('normal', df_normal_no_pca), ('cancer', df_cancer_no_pca)]:
                    if df_orig_no_pca is not None and not df_orig_no_pca.empty:
                        df_aligned_for_transform = df_orig_no_pca.reindex(columns=all_cols)
                        if not num_cols_for_pca_imputation.empty:
                            # 주의: transform 시에는 학습된 imputer_pca를 사용해야 합니다.
                            df_aligned_for_transform[num_cols_for_pca_imputation] = imputer_pca.transform(df_aligned_for_transform[num_cols_for_pca_imputation])
                        
                        df_pca_transformed_values = pca.transform(df_aligned_for_transform)
                        df_pca_transformed = pd.DataFrame(df_pca_transformed_values,
                                                          index=df_orig_no_pca.index,
                                                          columns=[f"{omics_name}_PCA_PC{i+1}" for i in range(adjusted_n_components)])
                        final_processed_omics_data[omics_name][sample_cat] = df_pca_transformed
                        print(f"    {omics_name} {sample_cat} 데이터 PCA 변환 후 형상: {df_pca_transformed.shape}")
            except Exception as e_pca_fit_transform:
                print(f"    오류: {omics_name} PCA 학습 또는 변환 중 오류: {e_pca_fit_transform}. 이 오믹스는 PCA 없이 사용.")
                if df_normal_no_pca is not None: final_processed_omics_data[omics_name]['normal'] = df_normal_no_pca
                if df_cancer_no_pca is not None: final_processed_omics_data[omics_name]['cancer'] = df_cancer_no_pca
        else:
            if df_normal_no_pca is not None: final_processed_omics_data[omics_name]['normal'] = df_normal_no_pca
            if df_cancer_no_pca is not None: final_processed_omics_data[omics_name]['cancer'] = df_cancer_no_pca
            if n_pca_components is not None:
                print(f"--- {omics_name}: PCA 적용 대상 데이터 부족 또는 설정 오류로 PCA 적용 안 함 ---")
            else:
                print(f"--- {omics_name}: PCA 적용 설정 없음 ---")


    # 4. 최종 샘플 기반 데이터셋 구축 (환자별로 오믹스 특징만 취합)
    processed_data_by_patient_for_final_merge = {}
    all_patient_ids_from_omics = set()
    for omics_name_key in final_processed_omics_data:
        for sample_category_key in ['normal', 'cancer']:
            df_current = final_processed_omics_data[omics_name_key].get(sample_category_key)
            if df_current is not None and not df_current.empty:
                all_patient_ids_from_omics.update(df_current.index.tolist())
    
    print(f"\n--- 총 {len(all_patient_ids_from_omics)} 명의 환자(오믹스 기준)에 대한 최종 오믹스 특징 취합 시작 ---")

    for patient_id_str in all_patient_ids_from_omics:
        processed_data_by_patient_for_final_merge[patient_id_str] = {'normal': {}, 'cancer': {}}
        for omics_name_key in final_processed_omics_data:
            df_normal_omics = final_processed_omics_data[omics_name_key].get('normal')
            if df_normal_omics is not None and patient_id_str in df_normal_omics.index:
                processed_data_by_patient_for_final_merge[patient_id_str]['normal'].update(df_normal_omics.loc[patient_id_str].to_dict())

            df_cancer_omics = final_processed_omics_data[omics_name_key].get('cancer')
            if df_cancer_omics is not None and patient_id_str in df_cancer_omics.index:
                processed_data_by_patient_for_final_merge[patient_id_str]['cancer'].update(df_cancer_omics.loc[patient_id_str].to_dict())
        
    all_samples_list = []
    for patient_id_str, data_by_type in processed_data_by_patient_for_final_merge.items():
        if data_by_type['normal']:
            sample_dict_normal = {PATIENT_ID_COL: patient_id_str, 'target': 0, 'sample_id': f"{patient_id_str}_NORMAL"}
            sample_dict_normal.update(data_by_type['normal'])
            all_samples_list.append(sample_dict_normal)
        
        if data_by_type['cancer']:
            sample_dict_cancer = {PATIENT_ID_COL: patient_id_str, 'target': 1, 'sample_id': f"{patient_id_str}_CANCER"}
            sample_dict_cancer.update(data_by_type['cancer'])
            all_samples_list.append(sample_dict_cancer)

    if not all_samples_list:
        print("오류: 최종 샘플 데이터가 없습니다. 전처리 과정 및 입력 파일을 확인하십시오."); return

    final_samples_df = pd.DataFrame(all_samples_list)
    print(f"\n최종 병합된 샘플 데이터 형상 (오믹스 데이터만 포함): {final_samples_df.shape}")

    if final_samples_df.empty: print("오류: 병합 후 데이터프레임이 비어있습니다."); return
    if 'sample_id' in final_samples_df.columns:
        final_samples_df = final_samples_df.set_index('sample_id')
    else: print("치명적 오류: 'sample_id' 컬럼이 생성되지 않았습니다.") ; return
    if 'target' not in final_samples_df.columns: print("오류: 타겟 변수('target')가 최종 DF에 없습니다."); return

    y = final_samples_df['target']
    cols_to_drop_from_X = ['target']
    if PATIENT_ID_COL in final_samples_df.columns: cols_to_drop_from_X.append(PATIENT_ID_COL)
    X = final_samples_df.drop(columns=cols_to_drop_from_X, errors='ignore')

    print(f"X 생성 후 형상: {X.shape}, y 분포:\n{y.value_counts(normalize=True)}")

    # 최종 X에 대한 결측치 처리 (SimpleImputer - median)
    # 이 imputer_final은 최종 병합된 X에 대한 결측치 처리이므로, 별도로 저장하지 않아도 됩니다.
    # 만약 예측 시에도 새로운 입력 데이터가 전체 X 형태를 가질 것으로 예상되고,
    # 해당 데이터에 결측치가 있을 가능성이 있다면 이 imputer_final도 저장하는 것을 고려할 수 있습니다.
    # 하지만 현재 오믹스 전용 파이프라인에서는 각 오믹스별 개별 전처리가 핵심이므로 필수적이지는 않습니다.
    if X.isnull().values.any():
        print(f"\n최종 특징 행렬 X에 결측치 발견 (형상: {X.shape}). SimpleImputer (median)로 대치합니다.")
        numeric_cols = X.select_dtypes(include=np.number).columns
        non_numeric_cols = X.select_dtypes(exclude=np.number).columns

        if not numeric_cols.empty:
            imputer_final = SimpleImputer(strategy='median')
            X_numeric_imputed = pd.DataFrame(imputer_final.fit_transform(X[numeric_cols]), columns=numeric_cols, index=X.index)
            if not non_numeric_cols.empty:
                print(f"경고: 다음 비숫자형 컬럼은 imputation에서 제외됩니다: {list(non_numeric_cols)}")
                X = pd.concat([X_numeric_imputed, X[non_numeric_cols]], axis=1)[X.columns]
            else:
                X = X_numeric_imputed
            print(f"결측치 처리 후 X 형상: {X.shape}")
            if X.isnull().values.any(): print("경고: 최종 결측치 처리 후에도 NaN 존재. 확인 필요.")
        else:
            print("경고: 최종 X에 숫자형 컬럼이 없어 imputation을 건너<0xEB>니다.")
    else:
        print("\n최종 특징 행렬 X에 결측치가 없습니다.")

    # 데이터 저장 (파일명 변경)
    try:
        features_output_path = 'final_sample_based_features_omics_only_revised.csv'
        target_output_path = 'final_sample_based_target_omics_only_revised.csv'
        X.to_csv(features_output_path, index=True)
        y.to_csv(target_output_path, index=True, header=['target'])
        print(f"\n전처리된 샘플 기반 데이터 저장 완료: {features_output_path}, {target_output_path}")
    except Exception as e: print(f"데이터 저장 중 오류: {e}")
    print("\n===== 샘플 기반 '암 vs 정상' 전처리 파이프라인 (오믹스 데이터만 사용, 특징 유출 방지 버전) 종료 =====")

if __name__ == '__main__':
    main_preprocessing_revised_omics_only()