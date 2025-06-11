import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
import requests
import xml.etree.ElementTree as ET
from google.cloud import vision
from google.oauth2 import service_account
import urllib3

# SSL 경고 무시 (과제용 임시 조치)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

# Google Cloud Vision API 인증
google_creds = dict(st.secrets["google_cloud"])

# 핵심: private_key 내부 '\\n'을 실제 줄바꿈 문자 '\n'로 바꾸기
google_creds["private_key"] = google_creds["private_key"].replace("\\n", "\n")

credentials = service_account.Credentials.from_service_account_info(google_creds)
client = vision.ImageAnnotatorClient(credentials=credentials)


def process_pill_image(pil_image):
    img = np.array(pil_image.convert("L"))  # 흑백 변환
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    # 도형 추정
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape = "기타"
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) > 6:
            shape = "원형"
        elif len(approx) >= 4:
            shape = "타원형"

    # OCR (Google Vision API 사용)
    buffered = pil_image.convert("RGB")
    buffered.save("temp.png")
    with open("temp.png", "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        text = texts[0].description.strip().replace("\n", " ")
    else:
        text = ""

    return shape, text


# --- API 요청 함수 ---
@st.cache_data(show_spinner=False)
def search_pill(shape, print_code):
    API_KEY = st.secrets.get("drug_api_key") or "API_KEY_HERE"  # 여기에 API 키 삽입
    url = "http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getPillList03"
    params = {
        "serviceKey": API_KEY,
        "item_shape": shape,
        "print_front": print_code,
        "numOfRows": 5,
        "pageNo": 1,
    }
    # SSL 검증 무시 - 과제용 임시 조치
    res = requests.get(url, params=params, verify=False)
    if res.status_code == 200:
        root = ET.fromstring(res.text)
        return root.findall(".//item")
    else:
        return []


# --- 이미지 처리 및 결과 출력 ---
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
