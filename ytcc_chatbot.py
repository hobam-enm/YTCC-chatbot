# -*- coding: utf-8 -*-
# 💬 유튜브 댓글 분석 챗봇: 자연어 질문 처리, 데이터 수집 및 AI 답변 생성

import streamlit as st
import pandas as pd
import os
import re
import time
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timedelta, timezone
import requests
import json

# === 0. 페이지 설정 (반드시 모든 st.xxx 호출 중 가장 먼저 와야 함) ===
st.set_page_config(page_title="💬 유튜브 댓글 분석 챗봇", layout="wide")

# ===== 1. 기본 설정 및 유틸리티 함수 (ytccai_cloud.py 및 ytcc_chatbot.py 기반) =====

# 사용자 요청에 따라 Streamlit Secrets에서 API 키를 로드하도록 수정합니다.
try:
    # 🔑 Streamlit secrets에서 API 키 로드
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, AttributeError, FileNotFoundError):
    # secrets에 없는 경우 환경 변수 또는 빈 문자열 사용 (fallback)
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    if not GEMINI_API_KEY:
        st.warning("⚠️ Streamlit Secrets(`[secrets] GEMINI_API_KEY`) 또는 환경 변수가 설정되지 않았습니다. 챗봇 기능이 작동하지 않을 수 있습니다.")

# YouTube API 키도 필요하지만, 이 예제에서는 댓글 수집 로직을 단순화합니다.
# 실제 ytccai_cloud.py 에서는 build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)를 사용합니다.

BASE_DIR = "/tmp"; os.makedirs(BASE_DIR, exist_ok=True)
KST = timezone(timedelta(hours=9))

def now_kst(): 
    return datetime.now(tz=KST)

# Streamlit 재실행을 안전하게 처리하는 함수
def safe_rerun():
    """Streamlit Cloud 환경에서 발생하는 오류를 피하며 재실행을 시도합니다."""
    try:
        st.rerun()
    except:
        pass

# YouTube URL 파싱 함수
def parse_youtube_url(url: str) -> str:
    """YouTube URL에서 video ID를 추출합니다."""
    if "youtu.be" in url:
        return urlparse(url).path[1:]
    if "youtube.com" in url:
        query = parse_qs(urlparse(url).query)
        if 'v' in query:
            return query['v'][0]
    return ""

# 댓글 데이터를 가짜로 시뮬레이션하는 함수
# 실제로는 ytccai_cloud.py의 fetch_comments_to_csv 로직을 통해 데이터를 수집해야 합니다.
def mock_fetch_comments(video_id: str, count: int = 100) -> pd.DataFrame:
    """실제 API 호출 없이 가상의 댓글 데이터를 생성합니다. (API 호출 시뮬레이션)"""
    st.info(f"데이터 수집 중... (Video ID: {video_id}) - 실제로는 YouTube API가 호출되어야 합니다.")
    time.sleep(2) # API 호출 시간 시뮬레이션
    
    data = []
    # 분석에 필요한 최소한의 데이터를 포함합니다.
    keywords = ["신제품", "비추", "좋아요", "별로", "강추", "가격", "성능", "예쁘다", "궁금"]
    for i in range(count):
        comment_text = f"이 영상 {video_id} 관련 댓글입니다. {keywords[i % len(keywords)]}에 대한 의견이예요. 정말 {['좋아요', '별로예요', '괜찮네요'][i % 3]}."
        data.append({
            'comment_id': f'C_{i}',
            'text': comment_text,
            'like_count': i % 20 + 1,
            'author': f'User_{i % 10}',
            'published_at': (now_kst() - timedelta(hours=i)).isoformat()
        })
    df = pd.DataFrame(data)
    return df

# ===== 2. Gemini API 호출 로직 (챗봇의 핵심) =====

# 1) 자연어 질문을 분석 명령 JSON으로 파싱하는 모델
def parse_user_query_to_json(user_query: str, last_url: str):
    """
    사용자 질문과 마지막 URL을 기반으로 JSON 형태의 분석 명령을 추출합니다.
    """
    if not GEMINI_API_KEY:
        st.error("API 키가 설정되지 않아 파싱을 수행할 수 없습니다.")
        return None

    # 분석 명령 스키마 정의 (ytccai_cloud의 로직을 활용하기 위한 명령 구조)
    schema = {
        "type": "OBJECT",
        "properties": {
            "target_url": {
                "type": "STRING",
                "description": f"사용자가 언급한 YouTube 영상 URL. 언급이 없으면 'last_url' 값인 '{last_url}'을 사용하거나 비워둡니다."
            },
            "analysis_type": {
                "type": "STRING",
                "description": "분석 유형: 'SUMMARY' (전체 요약), 'KEYWORD_SEARCH' (키워드 검색 및 요약), 'SENTIMENT' (긍부정 분석 요청), 'TOPICS' (주제별 분류 요청) 중 하나"
            },
            "keywords": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "분석에 필요한 키워드 리스트 (KEYWORD_SEARCH의 경우). 콤마로 구분된 키워드를 리스트로 변환합니다."
            }
        },
        "required": ["analysis_type"]
    }

    system_prompt = (
        "당신은 사용자 요청을 유튜브 댓글 분석 시스템이 이해할 수 있는 JSON 명령어로 변환하는 AI 비서입니다. "
        "사용자의 질문을 분석하여 'analysis_request' 객체를 생성해야 합니다. "
        f"마지막으로 사용된 URL은 '{last_url}'입니다. 만약 사용자가 새로운 URL을 제공하지 않았다면 이 URL을 'target_url'에 채워 넣습니다. "
        "URL이 유효하지 않거나 명시되지 않았다면 'target_url'은 빈 문자열로 둡니다. "
        "항상 JSON 형식으로만 응답해야 하며, JSON 스키마를 준수해야 합니다."
    )
    
    # API 호출
    try:
        payload = {
            "contents": [{ "parts": [{ "text": user_query }] }],
            "systemInstruction": { "parts": [{ "text": system_prompt }] },
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema
            }
        }
        
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"
        
        # Exponential Backoff 적용
        for attempt in range(3):
            response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            if response.status_code == 200:
                result = response.json()
                json_string = result["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(json_string)
            elif response.status_code == 429 and attempt < 2:
                time.sleep(2 ** attempt)  # 1초, 2초 대기
                continue
            else:
                st.error(f"파싱 모델 API 호출 실패: 상태 코드 {response.status_code}")
                st.json(response.json())
                return None
    except Exception as e:
        st.error(f"파싱 모델 처리 중 오류 발생: {e}")
        return None

# 2) 분석 결과를 자연어 답변으로 합성하는 모델
def synthesize_response(original_query: str, analysis_data: str):
    """
    분석 데이터(댓글 텍스트, 통계 등)와 원래 질문을 바탕으로 최종 답변을 생성합니다.
    """
    if not GEMINI_API_KEY:
        return "죄송합니다. API 키가 설정되지 않아 답변을 생성할 수 없습니다."

    system_prompt = (
        "당신은 유튜브 댓글 분석 전문가이자 친절한 챗봇입니다. "
        "사용자의 질문과 제공된 댓글 분석 결과를 바탕으로 통찰력 있고 이해하기 쉬운 답변을 한국어로 제공해야 합니다. "
        "데이터를 그대로 나열하지 않고, 핵심 요약과 통계를 포함하여 자연스러운 문장으로 구성하세요. "
        "사용자가 지정한 URL의 댓글을 분석한 결과를 기반으로 답변합니다. "
        f"사용자의 원래 질문: '{original_query}'"
    )
    
    user_prompt = f"분석된 데이터 요약:\n\n{analysis_data}\n\n이 데이터를 기반으로 사용자 질문에 답변해주세요."

    # API 호출
    try:
        payload = {
            "contents": [{ "parts": [{ "text": user_prompt }] }],
            "systemInstruction": { "parts": [{ "text": system_prompt }] },
            "config": { "temperature": 0.7 }
        }
        
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"
        
        for attempt in range(3):
            response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            if response.status_code == 200:
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            elif response.status_code == 429 and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            else:
                return f"⚠️ 답변 합성 모델 API 호출 실패: 상태 코드 {response.status_code}"

    except Exception as e:
        return f"⚠️ 답변 합성 중 오류 발생: {e}"

# ===== 3. Streamlit UI 및 챗봇 로직 구현 =====

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.last_url = ""
    st.session_state.comments_df = None
    st.session_state.messages.append({"role": "assistant", "content": 
        "안녕하세요! 유튜브 댓글 분석 챗봇입니다. 🤖\n\n먼저 **분석하고 싶은 YouTube 영상 URL**을 입력해주세요. URL 입력 후 댓글에 대해 자유롭게 질문해주세요. (예: '신제품에 대한 반응이 어때?', '가장 좋아요를 많이 받은 댓글은 뭐야?')"
    })

# --- Chat Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input Handler ---
if prompt := st.chat_input("YouTube URL을 입력하거나, 분석할 내용을 질문하세요."):
    
    # 1. 사용자 질문 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AI가 질문을 이해하고 분석을 준비하는 중입니다..."):
            
            # 2. Gemini 파싱 (질문 -> JSON 명령어)
            parsed_command = parse_user_query_to_json(prompt, st.session_state.last_url)

            if parsed_command is None:
                st.error("질문 분석에 실패했습니다. 다시 시도해주세요.")
                st.session_state.messages.append({"role": "assistant", "content": "질문 분석에 실패했습니다. 다시 시도해주세요."})
                safe_rerun()

            target_url = parsed_command.get("target_url")
            analysis_type = parsed_command.get("analysis_type", "SUMMARY")
            keywords = parsed_command.get("keywords", [])
            
            # URL 유효성 검사 및 업데이트
            video_id = parse_youtube_url(target_url)
            
            # 3. 데이터 수집 단계
            df = st.session_state.comments_df
            
            if video_id:
                # 새로운 URL이 입력되었거나 URL이 변경된 경우
                if video_id != parse_youtube_url(st.session_state.last_url):
                    st.info(f"새로운 URL을 감지했습니다. 댓글 데이터를 수집합니다. (ID: {video_id})")
                    df = mock_fetch_comments(video_id, count=200) # 댓글 200개 가상 수집
                    st.session_state.comments_df = df
                    st.session_state.last_url = target_url
                else:
                    st.info(f"기존 URL ({st.session_state.last_url})의 데이터를 사용합니다.")

            elif not df:
                # URL도 없고 기존 데이터도 없는 경우
                response = "죄송합니다. 분석할 YouTube 영상 URL이 없거나 유효하지 않습니다. 먼저 URL을 입력해주세요."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                safe_rerun()
                
            if df is not None:
                # 4. 분석 데이터 가공
                st.info(f"총 {len(df)}개의 댓글 데이터를 바탕으로 요청하신 '{analysis_type}' 분석을 수행합니다.")
                analysis_data_str = ""
                
                # 분석 유형에 따른 데이터 가공 로직 (ytccai_cloud.py의 로직 기반)
                
                if analysis_type == "KEYWORD_SEARCH" and keywords:
                    # 키워드 검색
                    keywords_pattern = '|'.join(re.escape(k) for k in keywords)
                    filtered_df = df[df['text'].str.contains(keywords_pattern, case=False, na=False)].copy()
                    
                    if not filtered_df.empty:
                        # 댓글 10개와 통계 요약
                        top_comments = filtered_df.sort_values(by='like_count', ascending=False).head(10)
                        analysis_data_str += f"### 키워드 '{', '.join(keywords)}' 관련 댓글 ({len(filtered_df)}개 발견)\n"
                        analysis_data_str += filtered_df.describe(include='all').to_markdown() + "\n\n"
                        analysis_data_str += "#### 상위 댓글 10개:\n"
                        analysis_data_str += top_comments[['text', 'like_count']].to_markdown(index=False)
                    else:
                        analysis_data_str = f"키워드 '{', '.join(keywords)}'와 일치하는 댓글이 발견되지 않았습니다. 전체 댓글 요약으로 전환합니다."
                        analysis_type = "SUMMARY"

                if analysis_type == "SUMMARY":
                    # 전체 요약
                    total_comments = len(df)
                    unique_authors = df['author'].nunique()
                    
                    top_liked = df.sort_values(by='like_count', ascending=False).iloc[0]
                    
                    # 가장 최근 댓글 50개를 요약에 사용
                    sample_comments = df['text'].tail(50).str.cat(sep='\n---\n')
                    
                    analysis_data_str += "### 전체 댓글 분석 통계\n"
                    analysis_data_str += f"- 총 댓글 수: {total_comments}개\n"
                    analysis_data_str += f"- 고유 작성자 수: {unique_authors}명\n"
                    analysis_data_str += f"- 최다 좋아요 댓글: \"{top_liked['text'][:50]}...\" ({top_liked['like_count']}개)\n\n"
                    analysis_data_str += "#### Gemini가 분석할 최근 댓글 샘플 (50개):\n"
                    analysis_data_str += sample_comments
                    
                
                # 5. Gemini 답변 합성
                final_response = synthesize_response(prompt, analysis_data_str)
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                
            safe_rerun()

# --- 사이드바 및 디버깅 정보 (선택 사항) ---
with st.sidebar:
    st.header("⚙️ 챗봇 상태")
    st.caption("개발 및 디버깅 정보")
    
    st.markdown("---")
    st.subheader("마지막 분석 URL")
    st.code(st.session_state.last_url)

    st.subheader("수집된 댓글 수")
    if st.session_state.comments_df is not None:
        st.info(f"{len(st.session_state.comments_df)}개")
    else:
        st.info("데이터 없음")
        
    st.markdown("---")
    st.subheader("로컬 데이터 정리")
    if st.button("🗑️ 세션 초기화", type="secondary"):
        st.session_state.clear()
        safe_rerun()
