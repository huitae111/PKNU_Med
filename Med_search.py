import streamlit as st
from zeep import Client
from zeep.transports import Transport
from requests import Session
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
from google.cloud import vision
from google.oauth2 import service_account

# 스트림릿 기본 설정
st.set_page_config(page_title="약 모양 그리기 검색기", layout="centered")
st.title("💊 약 모양 그리기 검색기")

# --- 1. 약 모양 그리는 캔버스 ---
st.markdown("### 1. 약의 대략적인 모양과 식별 문자를 그려보세요")
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

# --- 2. Google Cloud Vision API 인증 설정 ---
google_creds = dict(st.secrets["google_cloud"])
google_creds["private_key"] = google_creds["private_key"].replace("\\n", "\n")
credentials = service_account.Credentials.from_service_account_info(google_creds)
vision_client = vision.ImageAnnotatorClient(credentials=credentials)


# --- 3. 이미지 처리 함수 (모양 분류 + 문자 추출) ---
def process_pill_image(pil_image):
    # 모양 감지 (단순 윤곽선 기반)
    img = np.array(pil_image.convert("L"))
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape = "기타"
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) > 6:
            shape = "1"  # 원형
        elif len(approx) >= 4:
            shape = "2"  # 타원형

    # Vision API를 이용한 문자 인식
    buffered = pil_image.convert("RGB")
    buffered.save("temp.png")
    with open("temp.png", "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations
    text = texts[0].description.strip().replace("\n", " ") if texts else ""
    return shape, text


# --- 4. 식약처 낱알 식별 API 호출 ---
@st.cache_data
def search_pill_soap(item_shape, print_front):
    API_KEY = st.secrets["drug_api_key"]
    wsdl = "http://apis.data.go.kr/1471000/PillInfoService/getPillList?wsdl"
    session = Session()
    session.verify = False
    transport = Transport(session=session)

    try:
        client = Client(wsdl=wsdl, transport=transport)
        response = client.service.getPillList(
            serviceKey=API_KEY,
            item_shape=item_shape,
            print_front=print_front,
            numOfRows=5,
            pageNo=1,
        )
        return response
    except Exception as e:
        st.error(f"API 호출 에러: {e}")
        return None


# --- 5. 버튼 누르면 실행 ---
if st.button("🔍 약 정보 검색하기"):
    if canvas_result.image_data is not None:
        image = Image.fromarray(canvas_result.image_data[:, :, :3].astype(np.uint8))
        shape, code = process_pill_image(image)

        st.subheader("📌 추정 결과")
        st.write(f"- 추정된 모양: **{shape}**")
        st.write(f"- 추출된 문자: **{code}**")

        st.subheader("📋 약 정보 검색 결과")
        res = search_pill_soap(shape, code)
        if res and hasattr(res, "item"):
            for item in res.item:
                name = getattr(item, "item_name", "정보 없음")
                entp = getattr(item, "entp_name", "정보 없음")
                img_url = getattr(item, "item_image", None)
                st.markdown(f"### {name}")
                st.write(f"제약사: {entp}")
                if img_url:
                    st.image(img_url, width=120)
        else:
            st.warning("검색 결과가 없습니다. 좀 더 정확히 그려보세요!")
    else:
        st.warning("그림을 먼저 그려주세요!")
