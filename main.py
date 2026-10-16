import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="서울 기온 예측기", page_icon="🌡️", layout="wide"
)

st.title("🌡️ 서울 연도별 평균기온 분석 및 예측기")
st.write(
    "서울의 기온 데이터를 바탕으로 연도별 평균기온을 구하고, 회귀 직선을 통해 미래의 기온을 예측합니다."
)


@st.cache_data
def load_data():
  url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
  df = pd.read_csv(url, encoding="utf-8")
  return df


try:
  df = load_data()
except Exception as e:
  st.error(f"데이터를 불러오는 데 실패했습니다: {e}")
  st.stop()

# 데이터 전처리
# '날짜' 열을 datetime 형식으로 변환
df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

# 2025년까지의 데이터만 필터링 (수업 기준 기간)
df = df[df["날짜"].dt.year <= 2025]

# 연도 추출
df["연도"] = df["날짜"].dt.year

# 연도별 데이터 수(관측일 수)와 평균기온 계산
yearly_stats = (
    df.groupby("연도")
    .agg(관측일수=("평균기온", "count"), 연평균기온=("평균기온", "mean"))
    .reset_index()
)

# 관측일이 300일 이상인 해만 남기기
filtered_df = yearly_stats[yearly_stats["관측일수"] >= 300].copy()

# 회귀 분석 모델 학습
X = filtered_df[["연도"]]
y = filtered_df["연평균기온"]

model = LinearRegression()
model.fit(X, y)

filtered_df["예측기온"] = model.predict(X)

# 상관계수 계산
correlation = filtered_df["연도"].corr(filtered_df["연평균기온"])

# 기본 정보 표시
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
  st.metric(
      label="분석에 사용된 해의 개수", value=f"{len(filtered_df)}개"
  )
with col2:
  st.metric(
      label="시작 연도", value=int(filtered_df["연도"].min())
  )
with col3:
  st.metric(label="끝 연도", value=int(filtered_df["연도"].max()))

st.info(
    f"📊 **연도와 연평균기온의 상관계수:** `{correlation:.4f}` (1에 가까울수록 강한"
    " 양의 상관관계)"
)

# Plotly 그래프 작성
fig = px.scatter(
    filtered_df,
    x="연도",
    y="연평균기온",
    title="서울 연도별 평균기온 및 추세선",
    labels={"연도": "연도", "연평균기온": "연평균기온 (°C)"},
    template="plotly_white",
)

# 회귀 직선 추가
fig.add_trace(
    go.Scatter(
        x=filtered_df["연도"],
        y=filtered_df["예측기온"],
        mode="lines",
        name="회귀 직선",
        line=dict(color="red", width=2),
    )
)

st.plotly_chart(fig, use_container_width=True)

# 슬라이더를 통한 연도별 예상 기온 확인
st.markdown("---")
st.subheader("🔮 특정 연도 기온 예측하기")
selected_year = st.slider(
    "예측할 연도를 선택하세요 (1900 ~ 2100)",
    min_value=1900,
    max_value=2100,
    value=2026,
)

# 선택한 연도의 예상 기온 계산
predicted_temp = model.predict([[selected_year]])[0]

# 크게 표시
st.markdown(
    f"<h2 style='text-align: center; color: #2E86C1;'>🗓️ {selected_year}년 서울의"
    f" 예상 연평균기온</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<h1 style='text-align: center; color: #E74C3C;'>{predicted_temp:.2f}"
    " °C</h1>",
    unsafe_allow_html=True,
)
