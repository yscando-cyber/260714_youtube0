import os
import re
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from wordcloud import WordCloud

# ----------------------------------------------------------------
# 페이지 설정
# ----------------------------------------------------------------
st.set_page_config(page_title="🎬 유튜브 댓글 분석기", page_icon="💬", layout="wide")
st.title("💬 유튜브 댓글 분석 & 워드클라우드")
st.caption("YouTube Data API v3 기반 댓글 수집 · 키워드 분석 · 워드클라우드 시각화")

# ----------------------------------------------------------------
# API 키 불러오기 (Streamlit Secrets 우선, 환경변수 보조)
# ----------------------------------------------------------------
API_KEY = st.secrets.get("YOUTUBE_API_KEY", os.environ.get("YOUTUBE_API_KEY", ""))

if not API_KEY:
    st.error(
        "YOUTUBE_API_KEY가 설정되어 있지 않습니다.\n\n"
        "Streamlit Cloud → App settings → Secrets 에 아래와 같이 추가해주세요:\n\n"
        'YOUTUBE_API_KEY = "여기에_발급받은_키_입력"'
    )
    st.stop()

# ----------------------------------------------------------------
# 한국어 폰트 준비 (최초 1회 다운로드 후 캐시)
# ----------------------------------------------------------------
FONT_PATH = "NanumGothic-Regular.ttf"
FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Regular.ttf"


@st.cache_resource
def ensure_font():
    if not os.path.exists(FONT_PATH):
        try:
            urllib.request.urlretrieve(FONT_URL, FONT_PATH)
        except Exception:
            return None
    return FONT_PATH


font_path = ensure_font()

# ----------------------------------------------------------------
# 유틸 함수
# ----------------------------------------------------------------
def extract_video_id(text: str) -> str:
    text = text.strip()
    patterns = [
        r"(?:v=|/videos/|embed/|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", text):
        return text
    return ""


KOREAN_STOPWORDS = set("""
이 그 저 것 수 들 등 및 을 를 은 는 이 가 의 에 에서 와 과 도 로 으로
한 하다 있다 없다 이다 되다 아 어 음 요 죠 네 좀 진짜 정말 그냥 너무
그리고 그런데 하지만 그래서 그렇지만 근데 이거 저거 거 게 좋다 나 저
우리 제가 제 내가 내 너 당신 이번 저번 오늘 진심 완전 존나 그건 이건
""".split())

ENGLISH_STOPWORDS = set("""
the a an is are was were be been being to of and or in on at for with
this that these those it its i you he she they we my your his her our
their as not no so but if then just very really
""".split())


def clean_and_tokenize(text: str):
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
    tokens = re.findall(r"[가-힣]{2,}|[A-Za-z]{3,}", text)
    result = []
    for tok in tokens:
        low = tok.lower()
        if low in ENGLISH_STOPWORDS:
            continue
        if tok in KOREAN_STOPWORDS:
            continue
        result.append(tok)
    return result


@st.cache_data(ttl=600, show_spinner=False)
def fetch_video_info(video_id: str, api_key: str):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {"part": "snippet,statistics", "id": video_id, "key": api_key}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data.get("items"):
        return None
    item = data["items"][0]
    return {
        "title": item["snippet"]["title"],
        "channel": item["snippet"]["channelTitle"],
        "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
        "view_count": int(item["statistics"].get("viewCount", 0)),
        "like_count": int(item["statistics"].get("likeCount", 0)),
        "comment_count": int(item["statistics"].get("commentCount", 0)),
    }


@st.cache_data(ttl=600, show_spinner=False)
def fetch_comments(video_id: str, api_key: str, max_comments: int):
    comments = []
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    page_token = None

    while len(comments) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "key": api_key,
            "maxResults": 100,
            "order": "relevance",
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token

        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            break
        data = r.json()

        for item in data.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "author": top.get("authorDisplayName", "익명"),
                "text": top.get("textDisplay", ""),
                "likes": top.get("likeCount", 0),
                "published": top.get("publishedAt", ""),
                "reply_count": item["snippet"].get("totalReplyCount", 0),
            })
            if len(comments) >= max_comments:
                break

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return pd.DataFrame(comments)


def make_wordcloud(freq: dict, font_path: str):
    size = 900
    mask_img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(mask_img)
    draw.ellipse((15, 15, size - 15, size - 15), fill=0)
    mask = np.array(mask_img)

    wc = WordCloud(
        font_path=font_path,
        background_color="white",
        width=size,
        height=size,
        mask=mask,
        max_words=150,
        colormap="viridis",
        contour_width=2,
        contour_color="#4c78a8",
        prefer_horizontal=0.9,
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_alpha(0)
    return fig


# ----------------------------------------------------------------
# 사이드바 입력
# ----------------------------------------------------------------
st.sidebar.header("⚙️ 설정")
video_input = st.sidebar.text_input("유튜브 영상 URL 또는 ID", placeholder="https://www.youtube.com/watch?v=...")
max_comments = st.sidebar.slider("가져올 댓글 수", min_value=100, max_value=2000, value=500, step=100)
top_n_keywords = st.sidebar.slider("상위 키워드 개수", min_value=5, max_value=30, value=15)
analyze_btn = st.sidebar.button("🔍 분석 시작", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("YOUTUBE_API_KEY는 Streamlit Secrets에서 자동으로 불러옵니다.")

# ----------------------------------------------------------------
# 메인 로직
# ----------------------------------------------------------------
if analyze_btn:
    video_id = extract_video_id(video_input)

    if not video_id:
        st.error("유효한 유튜브 URL 또는 영상 ID를 입력해주세요.")
        st.stop()

    with st.spinner("영상 정보를 불러오는 중..."):
        try:
            info = fetch_video_info(video_id, API_KEY)
        except Exception as e:
            st.error(f"영상 정보를 불러오지 못했습니다: {e}")
            st.stop()

    if info is None:
        st.error("해당 영상을 찾을 수 없습니다. URL/ID를 확인해주세요.")
        st.stop()

    col_thumb, col_info = st.columns([1, 3])
    with col_thumb:
        st.image(info["thumbnail"], use_container_width=True)
    with col_info:
        st.subheader(info["title"])
        st.write(f"📺 채널: **{info['channel']}**")
        m1, m2, m3 = st.columns(3)
        m1.metric("조회수", f"{info['view_count']:,}")
        m2.metric("좋아요", f"{info['like_count']:,}")
        m3.metric("전체 댓글수", f"{info['comment_count']:,}")

    st.divider()

    with st.spinner(f"댓글 최대 {max_comments}개를 수집하는 중..."):
        try:
            df = fetch_comments(video_id, API_KEY, max_comments)
        except Exception as e:
            st.error(f"댓글을 불러오지 못했습니다: {e}")
            st.stop()

    if df.empty:
        st.warning("수집된 댓글이 없습니다. (댓글이 비활성화된 영상일 수 있습니다)")
        st.stop()

    st.success(f"✅ 댓글 {len(df):,}개 수집 완료!")

    # ------------------------------------------------------------
    # 핵심 지표
    # ------------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("수집된 댓글 수", f"{len(df):,}")
    c2.metric("평균 좋아요 수", f"{df['likes'].mean():.1f}")
    c3.metric("평균 답글 수", f"{df['reply_count'].mean():.1f}")

    # ------------------------------------------------------------
    # 워드클라우드 + 키워드 분석
    # ------------------------------------------------------------
    st.subheader("☁️ 댓글 워드클라우드")

    all_tokens = []
    for text in df["text"]:
        all_tokens.extend(clean_and_tokenize(str(text)))

    if not all_tokens:
        st.info("워드클라우드를 만들 만큼 충분한 텍스트가 없습니다.")
    else:
        freq = Counter(all_tokens)
        wc_col, bar_col = st.columns([1.2, 1])

        with wc_col:
            if font_path:
                fig = make_wordcloud(dict(freq), font_path)
                st.pyplot(fig, use_container_width=True)
            else:
                st.warning("한글 폰트를 불러오지 못해 워드클라우드를 표시할 수 없습니다.")

        with bar_col:
            top_items = freq.most_common(top_n_keywords)
            top_df = pd.DataFrame(top_items, columns=["키워드", "빈도"]).sort_values("빈도")
            bar_fig = px.bar(
                top_df, x="빈도", y="키워드", orientation="h",
                color="빈도", color_continuous_scale="viridis",
                title="상위 키워드 빈도",
            )
            bar_fig.update_layout(height=650, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(bar_fig, use_container_width=True)

    # ------------------------------------------------------------
    # 인기 댓글 Top 10
    # ------------------------------------------------------------
    st.subheader("🔥 좋아요 많은 댓글 Top 10")
    top_comments = df.sort_values("likes", ascending=False).head(10)
    for _, row in top_comments.iterrows():
        st.markdown(
            f"**{row['author']}** · 👍 {row['likes']:,} · 💬 답글 {row['reply_count']}\n\n"
            f"> {row['text']}"
        )
        st.markdown("---")

    # ------------------------------------------------------------
    # 원본 데이터
    # ------------------------------------------------------------
    with st.expander("📋 전체 댓글 데이터 보기"):
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("CSV로 다운로드", data=csv, file_name=f"{video_id}_comments.csv", mime="text/csv")

else:
    st.info("👈 왼쪽 사이드바에 유튜브 영상 URL을 입력하고 '분석 시작' 버튼을 눌러주세요.")
