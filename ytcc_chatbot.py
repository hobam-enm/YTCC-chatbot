# -*- coding: utf-8 -*-
# 💬 유튜브 댓글 분석 챗봇 — 완성본
# - 자연어 한 줄 → (기간/키워드/옵션) 해석(제미나이) → 영상 수집(YouTube API) → 댓글 수집(CSV 스트리밍) → AI 요약 + 정량 시각화
# - 정량 분석 결과는 사이드바로 분리하여 출력

import streamlit as st
import pandas as pd
import os, re, gc, time, json
import requests # <-- 추가: Gemini API 호출용
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from uuid import uuid4

# Google APIs
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai

# Visualization and NLP
import plotly.express as px
from plotly import graph_objects as go
import circlify
import numpy as np
from kiwipiepy import Kiwi
import stopwordsiso as stopwords

# ============== 0) 페이지/기본 ==============\nst.set_page_config(page_title="💬 유튜브 댓글 분석 챗봇", layout="wide")
st.title("💬 유튜브 댓글 분석 챗봇")

BASE_DIR = "/tmp"; os.makedirs(BASE_DIR, exist_ok=True)
KST = timezone(timedelta(hours=9))
# 챗봇 코드가 KST 시간대 처리를 위해 필요한 함수들
def now_kst(): return datetime.now(tz=KST)
def to_iso_kst(dt: datetime) -> str:
    if dt.tzinfo is None: dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST).isoformat(timespec="seconds")
def kst_to_rfc3339_utc(dt_kst: datetime) -> str:
    if dt_kst.tzinfo is None: dt_kst = dt_kst.replace(tzinfo=KST)
    return dt_kst.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
def parse_youtube_url(url: str) -> str:
    if "youtu.be" in url: return urlparse(url).path[1:]
    if "youtube.com" in url:
        query = parse_qs(urlparse(url).query)
        if 'v' in query: return query['v'][0]
    return ""
def safe_rerun():
    try: st.rerun()
    except: pass


# ============== 1) API Key 관리 ==============

@st.cache_resource
def _get_available_keys(key_name: str, default_fallback: list) -> list:
    """Streamlit Secrets에서 API 키 리스트를 로드합니다. 실패 시 빈 리스트 반환."""
    try:
        # Secrets에서 리스트 형태로 키를 로드
        keys = st.secrets.get(key_name, default_fallback)
        if not isinstance(keys, list):
            # 단일 키 문자열로 저장된 경우 리스트로 변환 시도
            if isinstance(keys, str) and keys:
                return [keys]
            raise ValueError(f"Secrets의 '{key_name}' 값이 리스트 형식이 아닙니다.")
        # 사용자가 제공한 키 목록을 secrets에서 로드하도록 설정 (AIzaSy로 시작하는 유효 키만)
        return [k for k in keys if k and k.startswith('AIzaSy')]
    except Exception as e:
        # st.secrets 접근 실패 시 (로컬 환경 등)
        print(f"[{key_name}] Secrets 로딩 실패: {e}")
        return default_fallback

# --- API Keys 로딩 (Secrets/환경 변수 우선) ---
# 기존 로직을 유지하여 Secrets에서 리스트 형태로 로드합니다.
YT_API_KEYS = _get_available_keys("YT_API_KEYS", [])
GEMINI_API_KEYS = _get_available_keys("GEMINI_API_KEYS", [])

if not YT_API_KEYS:
    st.error("🚨 YouTube API Key를 secrets.toml에 'YT_API_KEYS' 리스트 형태로 설정해주세요.")
if not GEMINI_API_KEYS:
    st.error("🚨 Gemini API Key를 secrets.toml에 'GEMINI_API_KEYS' 리스트 형태로 설정해주세요.")

# Kiwi 객체는 무거우므로 캐시 리소스로 관리
@st.cache_resource
def get_kiwi():
    return Kiwi(model_type='sbg', space_in_eojeol=True)

# 기타 상태 초기화
if "last_video_id" not in st.session_state: st.session_state.last_video_id = ""
if "last_comments_file" not in st.session_state: st.session_state.last_comments_file = None
if "df_for_sidebar" not in st.session_state: st.session_state.df_for_sidebar = None # 사이드바 시각화용 필터링된 DF
if "last_video_info" not in st.session_state: st.session_state.last_video_info = None # 사이드바 영상 정보용
if "last_schema" not in st.session_state: st.session_state.last_schema = None
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 🤖 **분석할 YouTube 영상 URL**을 입력하거나, 분석하고 싶은 주제를 질문해주세요. (예: '신제품 리뷰 영상 댓글 1주일치 요약해줘')"}]

# ============== 2) YouTube API 로직 (ytccai_cloud 로직 기반) ==============

def _get_youtube_service(key: str):
    """YouTube API 서비스 빌드."""
    try: return build('youtube', 'v3', developerKey=key, cache_discovery=False)
    except Exception: return None

def _rotate_key(api_keys: list) -> str:
    """사용 가능한 API 키 중 하나를 순환하여 반환."""
    if not api_keys: return None
    key = api_keys.pop(0)
    api_keys.append(key)
    return key

def get_video_info(video_id: str):
    """비디오 제목, 채널, 게시일 등의 기본 정보를 가져옵니다."""
    api_keys = YT_API_KEYS[:]
    for _ in range(len(api_keys)):
        key = _rotate_key(api_keys)
        if not key: continue
        try:
            youtube = _get_youtube_service(key)
            if not youtube: continue
            
            response = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=video_id
            ).execute()
            
            if response.get("items"):
                item = response["items"][0]
                published_at_utc = datetime.fromisoformat(item["snippet"]["publishedAt"].replace('Z', '+00:00'))
                return {
                    "title": item["snippet"]["title"],
                    "channelTitle": item["snippet"]["channelTitle"],
                    "published_at_kst": to_iso_kst(published_at_utc.astimezone(KST)),
                    "viewCount": int(item["statistics"].get("viewCount", 0)),
                    "likeCount": int(item["statistics"].get("likeCount", 0)),
                    "commentCount": int(item["statistics"].get("commentCount", 0))
                }
        except HttpError as e:
            if 'quotaExceeded' in str(e) or 'keyInvalid' in str(e):
                st.warning(f"YouTube API 키 할당량 초과 또는 무효: {key[:10]}...")
                continue
            st.error(f"비디오 정보 로드 오류: {e}")
            break
        except Exception as e:
            st.error(f"비디오 정보 로드 중 예기치 않은 오류: {e}")
            break
    return None

def fetch_comments_to_csv(video_id: str, max_results: int = 10000) -> str or None:
    """
    YouTube API를 사용하여 댓글을 수집하고 CSV 파일로 스트리밍 저장합니다.
    (ytccai_cloud.py의 메모리 최적화 로직 반영)
    """
    
    # 1. 파일 경로 설정
    filename = os.path.join(BASE_DIR, f"comments_{video_id}_{now_kst().strftime('%Y%m%d%H%M%S')}.csv")
    
    # 2. API 호출 로직
    api_keys = YT_API_KEYS[:]
    total_comments = 0
    next_page_token = None
    
    try:
        with open(filename, 'w', encoding='utf-8-sig') as f:
            # CSV 헤더 작성 (ytccai_cloud.py의 컬럼 사용)
            header = "comment_id,author,published_at,text,like_count,reply_count,video_id,parent_id\n"
            f.write(header)
            
            # 수집 루프
            for _ in range(20): # 최대 20페이지 (약 1만 개)
                key = _rotate_key(api_keys)
                if not key:
                    st.error("모든 YouTube API 키가 할당량 초과 또는 만료되었습니다.")
                    break
                    
                youtube = _get_youtube_service(key)
                if not youtube: continue
                
                request = youtube.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    maxResults=50,
                    pageToken=next_page_token,
                    order="time"
                )
                response = request.execute()
                
                # 댓글 쓰기
                for item in response.get("items", []):
                    top_level_comment = item["snippet"]["topLevelComment"]["snippet"]
                    comment_data = {
                        "comment_id": item["snippet"]["topLevelComment"]["id"],
                        "author": top_level_comment["authorDisplayName"],
                        "published_at": top_level_comment["publishedAt"],
                        "text": top_level_comment["textDisplay"].replace('\n', ' ').replace('\r', ' '),
                        "like_count": top_level_comment["likeCount"],
                        "reply_count": item["snippet"].get("totalReplyCount", 0),
                        "video_id": video_id,
                        "parent_id": None
                    }
                    
                    # CSV 라인 생성
                    # 필드 값에 콤마가 포함될 수 있으므로, CSV 규격에 맞게 텍스트는 큰따옴표로 감싸는 것이 안전하나,
                    # 기존 로직을 유지하고 텍스트 내 큰따옴표만 이스케이프 처리합니다.
                    line_values = [str(comment_data[col]).replace('"', '""') for col in header.strip().split(',')]
                    # text 필드만은 큰따옴표로 감싸서 CSV 안전성 확보 (필수)
                    line_values[3] = f'"{line_values[3]}"' 
                    
                    f.write(','.join(line_values) + '\n')
                    total_comments += 1

                next_page_token = response.get('nextPageToken')
                if not next_page_token or total_comments >= max_results:
                    break
            
        return filename if total_comments > 0 else None
        
    except HttpError as e:
        if 'commentsDisabled' in str(e):
            st.error("⚠️ 이 영상은 댓글이 비활성화되어 있습니다.")
        else:
            st.error(f"댓글 수집 중 API 오류 발생: {e}")
        if os.path.exists(filename): os.remove(filename)
        return None
    except Exception as e:
        st.error(f"댓글 수집 중 예기치 않은 오류 발생: {e}")
        if os.path.exists(filename): os.remove(filename)
        return None

# ============== 3) Gemini 로직 ==============

def _rotate_gemini_key():
    """Gemini API 키 순환."""
    return _rotate_key(GEMINI_API_KEYS)

def parse_user_query_to_schema(user_query: str, last_video_id: str) -> dict:
    """
    사용자 질문을 분석하여 유튜브 분석에 필요한 JSON 스키마를 추출합니다.
    (ytcc_chatbot.py의 핵심 파싱 로직 반영)
    """
    key = _rotate_gemini_key()
    if not key:
        return {"error": "Gemini API 키가 없습니다."}

    # 분석 명령 스키마 정의 (ytcc_chatbot.py 로직 기반)
    schema = {
        "type": "OBJECT",
        "properties": {
            "video_id": {
                "type": "STRING",
                "description": f"사용자가 언급한 유튜브 영상 ID. 언급이 없으면 '{last_video_id}'을 사용합니다. URL이 아닌 ID 자체여야 합니다."
            },
            "analysis_type": {
                "type": "STRING",
                "description": "'FULL_ANALYSIS' (새로운 영상/기간 요청), 'CHAT_FOLLOW_UP' (기존 데이터 기반 추가 질문), 'SIMPLE_QA' (일반 질문) 중 하나."
            },
            "start_iso": {
                "type": "STRING",
                "description": "분석 시작 날짜 (ISO 8601 형식). 예: '2024-01-01T00:00:00+09:00'"
            },
            "end_iso": {
                "type": "STRING",
                "description": "분석 종료 날짜 (ISO 8601 형식). 예: '2024-03-31T23:59:59+09:00'"
            },
            "keywords": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "사용자가 특별히 찾으려는 키워드 리스트 (예: '가격', '성능', '버그')."
            },
            "entities": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "키워드 외에 감지된 보조 주제나 엔티티 (예: '갤럭시 S24', '인공지능')."
            }
        },
        "required": ["analysis_type"]
    }
    
    # 기본값 설정
    today = now_kst()
    a_month_ago = today - timedelta(days=30)
    
    current_schema = {
        "video_id": last_video_id,
        "start_iso": to_iso_kst(a_month_ago.replace(hour=0, minute=0, second=0, microsecond=0)),
        "end_iso": to_iso_kst(today.replace(hour=23, minute=59, second=59, microsecond=0)),
        "keywords": [],
        "entities": []
    }
    
    # 이전 대화 맥락 추가 (간결하게)
    # history_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]]) # 사용하지 않음
    
    system_prompt = (
        "당신은 사용자 요청을 유튜브 댓글 분석 시스템이 이해할 수 있는 JSON 명령어로 변환하는 AI입니다. "
        "사용자의 질문과 대화 맥락을 고려하여, 필요한 경우 비디오 ID, 기간, 키워드를 추출하여 제공된 JSON 스키마에 맞춥니다. "
        f"마지막으로 분석한 영상 ID는 '{last_video_id}'입니다. 현재 날짜는 {today.strftime('%Y년 %m월 %d일')}입니다. "
        f"새로운 URL이 감지되면 'video_id'를 업데이트하고, 분석 요청이면 'FULL_ANALYSIS'로 설정하세요. "
        "기존 데이터에 대한 추가 질문이면 'CHAT_FOLLOW_UP'로 설정하고, 일반적인 질문은 'SIMPLE_QA'로 설정합니다. "
        "기간 정보가 없으면 기본값으로 지난 30일을 사용합니다. 반드시 JSON 형식으로만 응답해야 합니다."
    )
    
    try:
        payload = {
            "contents": [{ "parts": [{ "text": user_query }] }],
            "systemInstruction": { "parts": [{ "text": system_prompt }] },
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema
            }
        }
        
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={key}"
        
        # Exponential Backoff
        for attempt in range(3):
            response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            if response.status_code == 200:
                json_string = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(json_string)
                # URL이 있다면 ID로 변환하여 할당
                if parsed.get("video_id"):
                    if 'youtube.com' in parsed['video_id'] or 'youtu.be' in parsed['video_id']:
                        parsed['video_id'] = parse_youtube_url(parsed['video_id'])
                return {**current_schema, **parsed}
            elif response.status_code == 429 and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            else:
                st.error(f"파싱 모델 API 호출 실패: 상태 코드 {response.status_code}")
                # st.json(response.json()) # 디버깅 정보 제거
                return {"error": f"API 호출 실패: {response.status_code}"}
    except Exception as e:
        st.error(f"파싱 모델 처리 중 오류 발생: {e}")
        return {"error": str(e)}

def synthesize_response(query: str, analysis_data: str, video_info: dict, schema: dict) -> str:
    """분석 데이터와 원래 질문을 바탕으로 최종 답변을 생성합니다."""
    key = _rotate_gemini_key()
    if not key: return "Gemini API 키가 없습니다."

    system_prompt = (
        "당신은 유튜브 댓글 분석 전문가이자 친절한 챗봇입니다. "
        "제공된 '분석 결과 데이터'와 '비디오 정보'를 바탕으로 사용자의 질문에 통찰력 있고 이해하기 쉬운 한국어 답변을 제공해야 합니다. "
        "데이터를 그대로 나열하지 않고, 핵심 요약과 통계를 포함하여 자연스러운 문장으로 구성하세요. "
        "만약 '분석 결과 데이터'가 비어 있다면, 비디오 정보만 사용하여 일반적인 답변을 제공하세요."
    )
    
    video_info_str = json.dumps(video_info, ensure_ascii=False, indent=2) if video_info else "N/A"
    
    user_prompt = (
        f"사용자의 질문: '{query}'\n"
        f"분석 요청 스키마: {json.dumps(schema, ensure_ascii=False)}\n"
        f"비디오 기본 정보:\n{video_info_str}\n"
        f"댓글 분석 결과 데이터:\n{analysis_data}\n\n"
        "이 정보를 기반으로 사용자 질문에 답변해주세요. 특히 키워드나 기간이 지정되었다면 그 결과를 강조하세요."
    )

    try:
        payload = {
            "contents": [{ "parts": [{ "text": user_prompt }] }],
            "systemInstruction": { "parts": [{ "text": system_prompt }] }
        }
        
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={key}"
        
        for attempt in range(3):
            response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            if response.status_code == 200:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif response.status_code == 429 and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            else:
                return f"⚠️ 답변 합성 모델 API 호출 실패: 상태 코드 {response.status_code}"
    except Exception as e:
        return f"⚠️ 답변 합성 중 오류 발생: {e}"

# ============== 4) 데이터 분석 및 시각화 로직 ==============

def get_filtered_comments_df(comments_file: str, schema: dict) -> pd.DataFrame:
    """댓글 CSV 파일을 읽고, 기간 및 키워드에 따라 필터링합니다."""
    if not comments_file or not os.path.exists(comments_file):
        return None
    
    # 1. CSV 로드 (메모리 사용 최소화 위해 ChunkSize를 10만으로)
    df_list = []
    try:
        # 'text' 컬럼을 문자열로 명확히 지정하여 오류 방지
        for chunk in pd.read_csv(comments_file, chunksize=100000, encoding='utf-8-sig', dtype={'comment_id': str, 'parent_id': str, 'text': str}):
            df_list.append(chunk)
        df = pd.concat(df_list, ignore_index=True)
    except Exception as e:
        st.error(f"CSV 파일 로드 오류: {e}")
        return None
    
    if df.empty: return df

    # 2. 기간 필터링
    start_dt = datetime.fromisoformat(schema['start_iso'])
    end_dt = datetime.fromisoformat(schema['end_iso'])
    
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
    # KST로 변환 후 필터링
    df['published_at_kst'] = df['published_at'].dt.tz_convert(KST)
    
    df_filtered = df[(df['published_at_kst'] >= start_dt) & (df['published_at_kst'] <= end_dt)]
    
    # 3. 키워드 필터링
    keywords = [k for k in schema.get('keywords', []) if k]
    if keywords:
        # `text`가 NaN인 경우를 처리하기 위해 `fillna('')`를 추가
        pattern = '|'.join(re.escape(k) for k in keywords)
        df_filtered = df_filtered[df_filtered['text'].fillna('').str.contains(pattern, case=False, na=False)]

    return df_filtered.reset_index(drop=True)

def generate_analysis_summary(df: pd.DataFrame, schema: dict) -> str:
    """필터링된 DataFrame을 기반으로 AI에게 전달할 분석 요약을 생성합니다."""
    if df.empty:
        return "필터링된 댓글 데이터가 없어 정량 분석을 수행할 수 없습니다. (기간 또는 키워드 불일치)"
    
    summary = f"**[분석 결과 요약]**\n"
    summary += f"- 분석 대상 댓글 수: {len(df)}개\n"
    summary += f"- 고유 작성자 수: {df['author'].nunique()}명\n"
    
    # 최다 좋아요 댓글
    top_liked = df.sort_values(by='like_count', ascending=False).head(1)
    if not top_liked.empty:
        summary += f"- 최다 좋아요 댓글 ({top_liked['like_count'].iloc[0]}개, 작성자: {top_liked['author'].iloc[0]}): \"{top_liked['text'].iloc[0][:100].replace('\n', ' ')}...\"\n"

    # 키워드/엔티티 빈도 분석 (명사 추출)
    kiwi = get_kiwi()
    stop_words = stopwords.stopwords(["ko"])
    
    all_text = ' '.join(df['text'].dropna())
    tokens = kiwi.tokenize(all_text)
    
    nouns = [t.form for t in tokens if t.tag.startswith('N') and len(t.form) > 1 and t.form not in stop_words]
    
    # 챗봇 질문의 키워드도 카운터에 포함하여 중요도를 높임
    user_keywords = [k for k in schema.get('keywords', []) if k]
    nouns.extend(user_keywords * 10) # 키워드 가중치 부여
    
    noun_counts = Counter(nouns).most_common(20)
    
    summary += "\n**[주요 명사 빈도 (Top 20)]**\n"
    summary += ', '.join([f"{n[0]}({n[1]})" for n in noun_counts]) + "\n"

    # 댓글 샘플 (AI 답변 맥락 유지를 위해 최근 10개)
    sample_comments = df['text'].tail(10).str.cat(sep='\n---END---\n').replace('\n', ' ')
    summary += "\n**[댓글 샘플 (최근 10개)]**\n"
    summary += sample_comments
    
    return summary

# 시각화 함수 (UI 구성 시 사용)
def generate_keyword_chart(df: pd.DataFrame, title: str):
    """키워드 버블 차트 생성 (Top 30 명사 기반)."""
    if df.empty: return
    kiwi = get_kiwi()
    stop_words = stopwords.stopwords(["ko"])
    
    all_text = ' '.join(df['text'].dropna())
    tokens = kiwi.tokenize(all_text)
    nouns = [t.form for t in tokens if t.tag.startswith('N') and len(t.form) > 1 and t.form not in stop_words]
    noun_counts = Counter(nouns).most_common(30)
    
    if not noun_counts: return
    
    data = pd.DataFrame(noun_counts, columns=['keyword', 'count'])
    
    # 버블 차트 라이브러리 (circlify) 사용
    circles = circlify.circlify(
        data['count'].tolist(),
        target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )
    
    fig = go.Figure()
    
    for circle, (keyword, count) in zip(circles, noun_counts):
        if circle.level == 1:
            fig.add_shape(type="circle",
                xref="x", yref="y",
                x0=circle.x - circle.r, y0=circle.y - circle.r,
                x1=circle.x + circle.r, y1=circle.y + circle.r,
                fillcolor=px.colors.qualitative.Plotly[noun_counts.index((keyword, count)) % len(px.colors.qualitative.Plotly)],
                line_color="rgba(0,0,0,0)",
            )
            
            fig.add_annotation(
                x=circle.x, y=circle.y,
                text=f"{keyword}<br>({count})",
                showarrow=False,
                font=dict(size=min(max(10, count * 0.05), 18), color="white"),
                align="center",
            )
            
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_quantitative_analysis(df_filtered: pd.DataFrame, video_info: dict, schema: dict):
    """
    사이드바에 정량 분석 결과, 필터링 정보 및 다운로드 버튼을 렌더링합니다.
    """
    if df_filtered is None or df_filtered.empty or schema is None:
        st.subheader("📊 정량 분석 대기 중")
        st.info("댓글을 분석하면 여기에 결과(차트, 통계)가 표시됩니다.")
        return

    st.subheader("📊 정량 분석 결과")
    
    # 1. 필터링 정보
    with st.expander("분석 대상 요약", expanded=True):
        if video_info:
            st.markdown(f"**영상 제목:** `{video_info['title'][:50]}...`")
            st.markdown(f"**채널:** `{video_info['channelTitle']}`")
            st.markdown(f"**원 댓글 수:** `{video_info.get('commentCount', 'N/A'):,}`개")
        
        st.markdown("---")
        st.markdown(f"**필터링 기간:** `{schema['start_iso'].split('T')[0]} ~ {schema['end_iso'].split('T')[0]}`")
        keywords = [k for k in schema.get('keywords', []) if k]
        st.markdown(f"**적용 키워드:** {', '.join(keywords) or '(전체)'}")
        st.markdown(f"**분석 대상 댓글 수:** **{len(df_filtered):,}**개")
    
    # 2. 키워드 차트
    generate_keyword_chart(df_filtered, "주요 키워드 버블")
    
    # 3. 다운로드 (필터링된 댓글)
    st.markdown("---")
    csv_filtered = df_filtered.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "⬇️ 필터링된 댓글 CSV 다운로드",
        data=csv_filtered,
        file_name=f"filtered_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# ============== 5) 메인 챗봇 루프 ==============

# --- Chat Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input Handler ---
if prompt := st.chat_input("YouTube URL 또는 분석 질문을 입력하세요."):
    
    # 1. 사용자 질문 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        # 2. Gemini 파싱 (질문 -> Schema 명령어)
        with st.spinner("AI가 질문을 이해하고 분석 스키마를 추출하는 중..."):
            parsed_schema = parse_user_query_to_schema(prompt, st.session_state.last_video_id)
            st.session_state.last_schema = parsed_schema # 스키마 저장 (사이드바 출력용)

        if parsed_schema.get("error"):
            st.error(f"파싱 오류: {parsed_schema['error']}")
            st.session_state.messages.append({"role": "assistant", "content": f"죄송합니다. 질문 분석 중 오류가 발생했습니다: {parsed_schema['error']}"})
            safe_rerun()

        video_id = parsed_schema.get("video_id", "")
        analysis_type = parsed_schema.get("analysis_type", "SIMPLE_QA")
        
        # 3. 데이터 처리 및 AI 답변 생성
        
        df_filtered = None
        video_info = st.session_state.last_video_info # 기존 정보 사용 시도
        
        if analysis_type == "SIMPLE_QA":
            # 일반 QA만 요청받은 경우
            response = synthesize_response(prompt, "N/A", video_info, parsed_schema)
            st.markdown(response)
        
        elif analysis_type == "FULL_ANALYSIS":
            
            # --- 3-1. 비디오 정보 및 데이터 수집 ---
            if not video_id:
                st.warning("분석할 영상 ID를 찾을 수 없습니다. URL을 포함하여 다시 질문해주세요.")
                st.session_state.messages.append({"role": "assistant", "content": "분석할 영상 ID를 찾을 수 없습니다. URL을 포함하여 다시 질문해주세요."})
                safe_rerun()
            
            with st.spinner(f"비디오 정보 로드 및 댓글 수집 시작 (ID: {video_id})..."):
                
                # 비디오 정보 로드 및 상태 저장
                video_info = get_video_info(video_id)
                if not video_info:
                    st.error("비디오 정보를 로드할 수 없습니다. ID 또는 API 키를 확인해주세요.")
                    st.session_state.messages.append({"role": "assistant", "content": "비디오 정보를 로드할 수 없습니다. ID 또는 API 키를 확인해주세요."})
                    safe_rerun()
                st.session_state.last_video_info = video_info # <-- 상태 저장
                
                # 댓글 수집 (새로운 영상인 경우)
                if video_id != st.session_state.last_video_id:
                    comments_file = fetch_comments_to_csv(video_id)
                    if not comments_file:
                        st.error("댓글을 수집할 수 없거나 댓글이 비활성화되어 있습니다.")
                        st.session_state.messages.append({"role": "assistant", "content": "댓글을 수집할 수 없거나 댓글이 비활성화되어 있습니다."})
                        safe_rerun()
                    
                    st.session_state.last_video_id = video_id
                    st.session_state.last_comments_file = comments_file
                else:
                    comments_file = st.session_state.last_comments_file
            
            # --- 3-2. 데이터 필터링 및 분석 ---
            with st.spinner("수집된 데이터를 필터링하고 분석하는 중..."):
                df_filtered = get_filtered_comments_df(comments_file, parsed_schema)
                st.session_state.df_for_sidebar = df_filtered # <-- 사이드바용 DF 저장
                analysis_data = generate_analysis_summary(df_filtered, parsed_schema)
            
            # --- 3-3. AI 답변 합성 (시각화는 사이드바에서 처리) ---
            with st.spinner("AI가 분석을 기반으로 답변을 생성하는 중..."):
                final_response = synthesize_response(prompt, analysis_data, video_info, parsed_schema)
                st.markdown(final_response)

        
        elif analysis_type == "CHAT_FOLLOW_UP":
            
            # --- 3-1. 기존 데이터 재활용 및 필터링 ---
            if not st.session_state.last_comments_file:
                st.warning("이전에 분석된 영상 데이터가 없습니다. URL과 함께 분석 요청을 먼저 해주세요.")
                st.session_state.messages.append({"role": "assistant", "content": "이전에 분석된 영상 데이터가 없습니다. URL과 함께 분석 요청을 먼저 해주세요."})
                safe_rerun()
                
            comments_file = st.session_state.last_comments_file
            video_info = st.session_state.last_video_info # 이전에 저장된 정보 사용

            with st.spinner("기존 데이터를 필터링하고 추가 분석하는 중..."):
                df_filtered = get_filtered_comments_df(comments_file, parsed_schema)
                st.session_state.df_for_sidebar = df_filtered # <-- 사이드바용 DF 저장
                analysis_data = generate_analysis_summary(df_filtered, parsed_schema)

            # --- 3-2. AI 답변 합성 ---
            with st.spinner("AI가 추가 분석을 기반으로 답변을 생성하는 중..."):
                final_response = synthesize_response(prompt, analysis_data, video_info, parsed_schema)
                st.markdown(final_response)
        
# ============== 6) 사이드바 렌더링 ==============

with st.sidebar:
    st.header("⚙️ 분석 상태 및 정량 결과")
    
    # 1. 정량 분석 결과
    # 사이드바에서 DF를 사용해 차트를 그립니다.
    render_quantitative_analysis(
        st.session_state.df_for_sidebar,
        st.session_state.last_video_info,
        st.session_state.last_schema
    )
    
    st.markdown("---")
    
    # 2. 키/세션 정보
    st.subheader("키/세션 정보")
    st.write(f"YT Keys: {len(YT_API_KEYS)}개 / Gemini Keys: {len(GEMINI_API_KEYS)}개")
    st.markdown(f"**Last Video ID:** `{st.session_state.last_video_id or '(없음)'}`")

    # 3. 분석 스키마 정보
    st.subheader("마지막 분석 스키마")
    if st.session_state.last_schema:
        s = st.session_state.last_schema
        st.markdown(f"- **유형:** `{s.get('analysis_type', 'N/A')}`")
        st.markdown(f"- **기간:** `{s.get('start_iso', 'N/A').split('T')[0]} ~ {s.get('end_iso', 'N/A').split('T')[0]}`")
        keywords = s.get('keywords', [])
        st.markdown(f"- **키워드:** {', '.join(keywords) or '(없음)'}")
    else:
        st.caption("파싱 대기 중...")
