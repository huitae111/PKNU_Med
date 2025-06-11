import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
import requests
import xml.etree.ElementTree as ET
import os
import json
from google.cloud import vision
import io

st.set_page_config(page_title="약 모양 그리기 검색기", layout="centered")
st.title("💊 약 모양 그리기 검색기")

# --- Canvas 설정 ---
st.markdown("""### 1. 약의 대략적인 모양과 식별 문자를 그려보세요""")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=5,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

# --- 구글 서비스 계정 인증 JSON 파일 만들기 ---
# secrets.toml에 아래처럼 google_cloud 키 안에 JSON 키 전체를 넣어야 함
# 예: st.secrets["google_cloud"]["private_key"], "client_email" 등
google_creds = st.secrets["google_cloud"]

# private_key는 줄바꿈 문자(\n)를 실제 줄바꿈으로 변경
google_creds["private_key"] = google_creds["private_key"].replace("\\n", "\n")

with open("service_account.json", "w") as f:
    json.dump(google_creds, f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"

# Vision API 클라이언트 생성
client = vision.ImageAnnotatorClient()

def process_pill_image(pil_image):
    # 흑백 변환 + 도형 추정
    img = np.array(pil_image.convert("L"))
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape = "기타"
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) > 6:
            shape = "원형"
        elif len(approx) >= 4:
            shape = "타원형"
        else:
            shape = "기타"

    # PIL 이미지를 바이트 형태로 변환 (Vision API 요청용)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    image_content = buffer.getvalue()

    image = vision.Image(content=image_content)
    response = client.text_detection(image=image)

    if response.error.message:
        st.error(f"Vision API 오류: {response.error.message}")
        return shape, ""

    texts = response.text_annotations
    text = texts[0].description.strip() if texts else ""

    return shape, text

@st.cache_data(show_spinner=False)
def search_pill(shape, print_code):
    API_KEY = st.secrets.get("drug_api_key") or "API_KEY_HERE"  # 여기에 API 키 삽입
    url = "https://apis.data.go.kr/1471000/DURPrdlstInfoService03/getPillList03"
    params = {
        "serviceKey": API_KEY,
        "item_shape": shape,
        "print_front": print_code,
        "numOfRows": 5,
        "pageNo": 1,
    }
    res = requests.get(url, params=params)
    if res.status_code == 200:
        root = ET.fromstring(res.text)
        return root.findall(".//item")
    else:
        return []

if st.button("🔍 약 정보 검색하기"):
    if canvas_result.image_data is not None:
        image = Image.fromarray((canvas_result.image_data[:, :, :3]).astype(np.uint8))
        shape, code = process_pill_image(image)

        st.subheader("📌 추정 결과")
        st.write(f"- 추정된 모양: **{shape}**")
        st.write(f"- 추출된 문자: **{code}**")

        st.subheader("📋 약 정보 검색 결과")
        items = search_pill(shape, code)
        if items:
            for item in items:
                name = item.findtext("item_name")
                entp = item.findtext("entp_name")
                img_url = item.findtext("item_image")
                st.markdown(f"### {name}")
                st.write(f"제약사: {entp}")
                if img_url:
                    st.image(img_url, width=120)
        else:
            st.warning("검색 결과가 없습니다. 좀 더 정확히 그려보세요!")
    else:
        st.warning("그림을 먼저 그려주세요!")
