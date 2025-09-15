import streamlit as st
import pandas as pd
import numpy as np
from .DCC import *
import warnings
from scipy.special import inv_boxcox

warnings.filterwarnings("ignore")

def max_corr(df: pd.DataFrame, target: str, corr_func=pearson_matrix) -> float:
    """
    Calculate the maximum correlation of a target column with other columns in the dataframe
    """
    corr_matrix = corr_func(df)
    if target not in corr_matrix.columns:
        raise ValueError(f"Target column {target} not found in dataframe")
    target_corr = corr_matrix[target].drop(target)  # Exclude self-correlation
    max_corr_value = target_corr.abs().max()
    return max_corr_value

def feature_corr(df: pd.DataFrame, target: str, corr_func=pearson_matrix) -> pd.Series:
    """
    Calculate the correlation of a target column with other columns in the dataframe
    """
    corr_matrix = corr_func(df)
    if target not in corr_matrix.columns:
        raise ValueError(f"Target column {target} not found in dataframe")
    target_corr = corr_matrix[target].drop(target)  # Exclude self-correlation
    return target_corr

def check_constant_columns(df: pd.DataFrame) -> list:
    """
    Check for constant columns in the dataframe
    """
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    return constant_cols

def run():
    st.set_page_config(layout="wide")
    st.title("🌟 Data Correlation Convergency")
    st.write("""
    This app allows you to upload a CSV file and evaluate the data quality of the dataset.
    """
    )

    st.header("📂 Upload your CSV data")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # all transform to float64
        df = df.select_dtypes(include=[np.number]).astype(np.float64)
        st.write("Data Preview:")
        st.dataframe(df.head())
        target_column = st.selectbox("Select the target column (if any)", options=[None] + list(df.columns))

    run_analysis = st.button("Run Data Quality Analysis", type="primary")



    if run_analysis:
        if uploaded_file is None:
            st.error("Please upload a CSV file to proceed.")
        elif target_column is None:
            st.error("Please select a target column to proceed.")
        elif target_column not in df.columns:
            st.error(f"The selected target column '{target_column}' is not in the dataframe.")
        else:
            if df.empty:
                st.error("The uploaded CSV file is empty.")
            else:
                # st.subheader("Data Quality Report")
                # st.write("Correlation Matrix:")
                # corr = df.corr()
                # st.plotly_chart(
                #     px.imshow(
                #         corr,
                #         text_auto=True,
                #         aspect="auto",
                #         color_continuous_scale="RdBu_r",
                #         title="Correlation Matrix",
                #     )
                # )
                st.header("📝DCC Analysis Results")
                constant_cols = check_constant_columns(df)
                if constant_cols:
                    st.warning(f"The following columns are constant and will be excluded from the analysis: {', '.join(constant_cols)}")
                    df = df.drop(columns=constant_cols)
                if df.shape[1] < 2:
                    st.error("The dataframe must contain at least two non-constant numeric columns for DCC analysis.")
                    return
                if target_column in constant_cols:
                    st.error("The target column cannot be a constant column.")
                    return
                st.badge("The default parameters for DCC: Perturbation ratio=0.05, ε=0.04, Pertubation type = Drop.")
                with st.spinner("Calculating DCC...", show_time=True):
                    dcc_pearson = repeated_avg_dcc(data=df, ratio=0.05, eps=0.04, corr_func=pearson_matrix, repeats=5)
                    dcc_spearman = repeated_avg_dcc(data=df, ratio=0.05, eps=0.04, corr_func=spearman_matrix, repeats=5)
                    dcc_kendall = repeated_avg_dcc(data=df, ratio=0.05, eps=0.04, corr_func=kendall_matrix, repeats=5)
                    dcc_js = repeated_avg_dcc(data=df, ratio=0.05, eps=0.04, corr_func=js_corr_matrix, repeats=5)
                    dcc_wd = repeated_avg_dcc(data=df, ratio=0.05, eps=0.04, corr_func=wd_corr_matrix, repeats=5)
                    dcc_xi = repeated_avg_dcc(data=df, ratio=0.05, eps=0.04, corr_func=xi_matrix, repeats=5)
                    dcc_mi = repeated_avg_dcc(data=df, ratio=0.05, eps=0.04, corr_func=mutual_info_matrix, repeats=5)
                    dcc_dc = repeated_avg_dcc(data=df, ratio=0.05, eps=0.04, corr_func=dcor_matrix, repeats=5)
                    
                    max_pearson = max_corr(df, target_column, corr_func=pearson_matrix)
                    max_js = max_corr(df, target_column, corr_func=js_corr_matrix)
                    max_mi = max_corr(df, target_column, corr_func=mutual_info_matrix)
                    max_dc = max_corr(df, target_column, corr_func=dcor_matrix)
                # st.success("DCC Calculation Complete!")
                # Display results as a table
                dcc_results = pd.DataFrame({
                    "Correlation Function": ["Pearson", "Spearman", "Kendall", "Jensen-Shannon", "Wasserstein Distance", "Chatterjee Xi Correlation", "Mutual Information", "Distance Correlation"],
                    "DCC Value": [dcc_pearson, dcc_spearman, dcc_kendall, dcc_js, dcc_wd, dcc_xi, dcc_mi, dcc_dc],
                })
                dcc_results = dcc_results.set_index("Correlation Function")
                st.table(dcc_results)
                import pickle
                score_pred_model = pickle.load(open("app/models/rf_model_scores.pkl", "rb"))
                dcc_input = np.array([[max_dc, max_js, max_pearson,dcc_dc, dcc_js, dcc_pearson]])
                quality_score = score_pred_model.predict(dcc_input)[0]
                quality_score = inv_boxcox(quality_score, 2.521146995683322)
                st.subheader("Predicted Model Performance")
                # st.write(f"The Predicted Model Performance: **{quality_score:.4f}**")
                st.write("The Predicted Model Performance (Accuracy/R²) is:")
                st.markdown(f'<span style="font-size:20px; font-weight:bold; color:white; background-color:#0099ff; padding:4px 8px; border-radius:4px;">{quality_score:.4f}</span>', unsafe_allow_html=True)

                # feature importance
                all_feats = df.columns.tolist()
                all_feats.remove(target_column)
                dcc_features_pearson = dcc_features(
                    data=df, feats=all_feats, ratio=0.1, eps=0.04, corr_func=pearson_matrix
                )
                dcc_features_js = dcc_features(
                    data=df, feats=all_feats, ratio=0.1, eps=0.04, corr_func=js_corr_matrix
                )
                dcc_features_mi = dcc_features(
                    data=df, feats=all_feats, ratio=0.1, eps=0.04, corr_func=mutual_info_matrix
                )
                dcc_features_dc = dcc_features(
                    data=df, feats=all_feats, ratio=0.1, eps=0.04, corr_func=dcor_matrix
                )
                feature_importance = pd.DataFrame({
                    "Feature": all_feats,
                    "DCC Pearson": [dcc_features_pearson[feat] for feat in all_feats],
                    "DCC JS": [dcc_features_js[feat] for feat in all_feats],
                    "DCC MI": [dcc_features_mi[feat] for feat in all_feats],
                    "DCC DC": [dcc_features_dc[feat] for feat in all_feats],
                })
                feature_importance = feature_importance.set_index("Feature")


                features_pearson = feature_corr(df, target_column, corr_func=pearson_matrix)
                features_js = feature_corr(df, target_column, corr_func=js_corr_matrix)
                features_mi = feature_corr(df, target_column, corr_func=mutual_info_matrix)
                features_dc = feature_corr(df, target_column, corr_func=dcor_matrix)
                feature_correlation = pd.DataFrame({
                    "Feature": features_pearson.index,
                    "Corr Pearson": features_pearson.values,
                    "Corr JS": features_js.values,
                    "Corr MI": features_mi.values,
                    "Corr DC": features_dc.values,
                })
                feature_correlation = feature_correlation.set_index("Feature")


                shap_pred_model = pickle.load(open("app/models/rf_model_shap.pkl", "rb"))
                feature_shapes = {}
                for feat in all_feats:
                    # ['Corr_pearson_matrix', 'Corr_mutual_info_matrix', 'Corr_js_corr_matrix', 'Corr_dcor_matrix', 'DCC_pearson_matrix', 'DCC_mutual_info_matrix', 'DCC_js_corr_matrix', 'DCC_dcor_matrix']
                    shap_input = np.array([[features_pearson[feat], features_mi[feat], features_js[feat], features_dc[feat],
                                            dcc_features_pearson[feat], dcc_features_mi[feat], dcc_features_js[feat], dcc_features_dc[feat]]])
                    shap_value = shap_pred_model.predict(shap_input)[0]
                    shap_value = inv_boxcox(shap_value, -0.012064615760812778)
                    feature_shapes[feat] = shap_value
                feature_shapes_df = pd.DataFrame.from_dict(feature_shapes, orient='index', columns=['Predicted SHAP Value'])
                feature_shapes_df = feature_shapes_df.sort_values(by='Predicted SHAP Value', ascending=False)

                # 设置索引为特征名
                feature_shapes_df.index.name = "Feature"
                feature_shapes_df.reset_index(inplace=True)
                feature_shapes_df = feature_shapes_df.set_index("Feature")

                st.subheader("Predicted Feature Importance (SHAP Values)")
                st.write("The predicted Feature Importance based on SHAP values is shown below. Higher SHAP values indicate greater importance of the feature in predicting the target variable.")
                # st.bar_chart(feature_shapes_df)
                st.table(feature_shapes_df)
                # st.subheader("DCC on features")
                # st.table(feature_importance)
                # st.subheader(f"Correlation with target: {target_column}")
                # st.table(feature_correlation)