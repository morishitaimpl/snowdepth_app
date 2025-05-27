import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pathlib

st.set_page_config(
    page_title="積雪深予測アプリ",
    page_icon="❄️",
    layout="wide"
)

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        with open("sapporo_models/snow_depth_model.pkl", "rb") as f:
            sapporo_model = pickle.load(f)
        
        with open("sapporo_models/scaler.pkl", "rb") as f:
            sapporo_scaler = pickle.load(f)
        
        with open("obihiro_models/snow_depth_model.pkl", "rb") as f:
            obihiro_model = pickle.load(f)
        
        with open("obihiro_models/scaler.pkl", "rb") as f:
            obihiro_scaler = pickle.load(f)

        return sapporo_model, sapporo_scaler, obihiro_model, obihiro_scaler
    except FileNotFoundError:
        return None, None, None, None

def predict_snow_depth(model, scaler, input_data):
    """Make prediction using the loaded model"""
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    return prediction[0]

def main():
    st.title("❄️ 積雪深予測アプリ")
    st.markdown("積雪深を予測します。")
    st.markdown("以下に気象データを入力してください")
    
    st.subheader("地域選択")
    region = st.selectbox(
        "地域を選択してください",
        ["札幌市", "帯広市"])
    
    sapporo_model, sapporo_scaler, obihiro_model, obihiro_scaler = load_model_and_scaler()
    
    if sapporo_model is None or obihiro_model is None:
        st.error("モデルが読み込めませんでした。まずモデルを訓練してください。")
        if st.button("モデルを訓練"):
            with st.spinner("モデルを訓練中..."):
                from train_model import train_and_save_model
                train_and_save_model("data_2016_2025.csv", "sapporo_models")
                train_and_save_model("data_2016_2025.csv", "obihiro_models")
                st.success("モデルの訓練が完了しました！ページを再読み込みしてください。")
        return
    
    if region == "札幌":
        model, scaler = sapporo_model, sapporo_scaler
    else:  # 帯広
        model, scaler = obihiro_model, obihiro_scaler
    
    st.subheader("気象データ入力")
    
    col1, col2 = st.columns(2)
    
    with col1:
        month = st.selectbox("月 (Month)", options=list(range(1, 13)), index=0)
        day = st.number_input("日 (Day)", min_value=1, max_value=31, value=1)
        land_atmosphere = st.number_input("陸上気圧 (Land Atmosphere)", value=1013.25, format="%.2f")
        sea_atmosphere = st.number_input("海上気圧 (Sea Atmosphere)", value=1013.25, format="%.2f")
        precipitation = st.number_input("降水量 (Precipitation)", min_value=0.0, value=0.0, format="%.1f")
        temperature = st.number_input("気温 (Temperature °C)", value=0.0, format="%.1f")
    
    with col2:
        humidity = st.number_input("湿度 (Humidity %)", min_value=0, max_value=100, value=50)
        wind_speed = st.number_input("風速 (Wind Speed)", min_value=0.0, value=5.0, format="%.1f")
        wind_direction = st.number_input("風向 (Wind Direction)", min_value=0.0, max_value=360.0, value=0.0, format="%.1f")
        snow_falling = st.number_input("降雪量 (Snow Falling)", min_value=0.0, value=0.0, format="%.1f")
        melted_snow = st.number_input("融雪量 (Melted Snow)", value=0.0, format="%.1f")
    
    if st.button("積雪深を予測", type="primary"):
        input_data = [
            month, day, land_atmosphere, sea_atmosphere, precipitation,
            temperature, humidity, wind_speed, wind_direction, snow_falling, melted_snow
        ]
        
        try:
            if temperature >= 10:
                prediction = 0
            else:
                prediction = predict_snow_depth(model, scaler, input_data)
            
            st.subheader("予測結果")
            st.metric("予測積雪深", f"{prediction:.1f} cm")
            
            st.subheader("入力データ確認")
            input_df = pd.DataFrame({
                "項目": ["月", "日", "陸上気圧", "海上気圧", "降水量", "気温", "湿度", "風速", "風向（角度）", "降雪量", "融雪量"],
                "値": [f"{val:.1f}" if isinstance(val, float) else str(val) for val in input_data],
                "単位": ["", "", "hPa", "hPa", "mm", "°C", "%", "m/s", "度", "mm", "mm"]
            })
            st.dataframe(input_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"予測中にエラーが発生しました: {str(e)}")
    
    with st.expander("ℹ️ アプリについて"):
        st.markdown("""
        このアプリは機械学習（ニューラルネットワーク）を使用して気象データから積雪深を予測します。
        
        **入力パラメータ:**
        - 月、日: 日付情報
        - 陸上気圧、海上気圧: 大気圧データ (hPa)
        - 降水量: 降水量 (mm)
        - 気温: 気温 (°C)
        - 湿度: 相対湿度 (%)
        - 風速: 風の速度 (m/s)
        - 風向: 風の方向 (度)
        - 降雪量: 新たな降雪量 (mm)
        - 融雪量: 融けた雪の量 (mm)
        
        **出力:** 予測積雪深 (cm)
        """)

if __name__ == "__main__":
    main()
