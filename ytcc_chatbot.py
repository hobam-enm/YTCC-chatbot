# -*- coding: utf-8 -*-
# 💬 유튜브 댓글 AI 요약 챗봇 (요청 준수: UI 개선, 정량 분석 완전 제거, 프롬프트/스키마 원본 복원)
# - 원본 파일의 "키워드 검색 및 다중 영상 댓글 수집" 로직 복원
# - 정량 분석 로직 (차트, 키워드 빈도, 형태소 분석) 완전히 제거
# - Gemini 스키마 (parse_user_query_to_schema) 원본 복잡 구조 그대로 복원

import streamlit as st
import pandas as pd
import os, re, gc, time, json
import requests
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from uuid import uuid4

# Google APIs
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# 정량 분석 관련 라이브러리 (plotly, kiwi, stopwordsiso, circlify 등)는 모두 제거됨

# ============== 0) 페이지/기본 설정 ==============
st.set_page_config(page_title="💬 유튜브 댓글 AI 요약 챗봇", layout="wide")

# UI 스타일링 (채팅 앱 느낌 강조)
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
}
.stChatMessage {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.stChatInput {
    border-top: 1px solid #e0e0e0;
    padding-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("💬 유튜브 댓글 AI 요약 챗봇")
st.markdown("**키워드(예: 신제품 리뷰) 또는 URL**을 입력하고, 기간/키워드와 함께 요약을 요청해 보세요. (예: `신제품 리뷰 영상 1주일치 요약해줘`)")
st.markdown("---")

BASE_DIR = "/tmp"; os.makedirs(BASE_DIR, exist_ok=True)
KST = timezone(timedelta(hours=9))

def now_kst(): return datetime.now(tz=KST)
def to_iso_kst(dt: datetime) -> str:
    if dt.tzinfo is None: dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST).isoformat(timespec="seconds")
def parse_youtube_url(url: str) -> str:
    if "youtu.be" in url: return urlparse(url).path[1:]
    if "youtube.com" in url:
        query = parse_qs(urlparse(url).query)
        if 'v' in query: return query['v'][0]
    return ""
def safe_rerun():
    """Streamlit의 Rerun을 안전하게 호출"""
    try: st.rerun()
    except: pass


# ============== 1) API Key 관리 ==============

@st.cache_resource
def _get_available_keys(key_name: str, default_fallback: list) -> list:
    """Streamlit Secrets에서 API 키 리스트를 로드합니다. 실패 시 빈 리스트 반환."""
    try:
        keys = st.secrets.get(key_name, default_fallback)
        if not isinstance(keys, list):
            if isinstance(keys, str) and keys: return [keys]
            raise ValueError(f"Secrets의 '{key_name}' 값이 리스트 형식이 아닙니다.")
        return [k for k in keys if k and k.startswith('AIzaSy')]
    except Exception:
        return default_fallback

# --- API Keys 로딩 (Secrets/환경 변수 우선) ---
YT_API_KEYS = _get_available_keys("YT_API_KEYS", [""])
GEMINI_API_KEYS = _get_available_keys("GEMINI_API_KEYS", [""])

if not YT_API_KEYS or YT_API_KEYS == [""]:
    st.sidebar.error("🚨 YouTube API Key를 설정해주세요.")
if not GEMINI_API_KEYS or GEMINI_API_KEYS == [""]:
    st.sidebar.error("🚨 Gemini API Key를 설정해주세요.")

# 상태 초기화
if "last_video_ids" not in st.session_state: st.session_state.last_video_ids = []
if "last_comments_file" not in st.session_state: st.session_state.last_comments_file = None
if "last_video_info_df" not in st.session_state: st.session_state.last_video_info_df = None 
if "last_schema" not in st.session_state: st.session_state.last_schema = None
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 🤖 **분석할 키워드 또는 YouTube 영상 URL**을 입력하거나, 분석하고 싶은 주제를 질문해주세요. (예: '신제품 리뷰 영상 댓글 1주일치 요약해줘')"}]

# ============== 2) YouTube API 로직 (다중 영상 처리 로직) ==============

def _get_youtube_service(key: str):
    """YouTube API 서비스 빌드."""
    try: 
        if not key: return None
        return build('youtube', 'v3', developerKey=key, cache_discovery=False)
    except Exception: return None

def _rotate_key(api_keys: list) -> str:
    """사용 가능한 API 키 중 하나를 순환하여 반환."""
    if not api_keys: return None
    key = api_keys.pop(0)
    api_keys.append(key)
    return key

def search_videos(query: str, max_videos: int = 5, published_after: datetime = None) -> (list, pd.DataFrame or None):
    """YouTube 검색 API를 통해 영상을 검색하고 기본 정보를 반환합니다."""
    api_keys = YT_API_KEYS[:]
    video_list = []
    
    # ISO 8601 형식 (UTC)
    published_after_utc = published_after.astimezone(timezone.utc).isoformat("T") + "Z" if published_after else None
    
    for _ in range(len(api_keys)):
        key = _rotate_key(api_keys)
        if not key: continue
        try:
            youtube = _get_youtube_service(key)
            if not youtube: continue
            
            response = youtube.search().list(
                q=query,
                part="snippet",
                type="video",
                maxResults=max_videos,
                publishedAfter=published_after_utc,
                order="relevance"
            ).execute()
            
            for item in response.get("items", []):
                video_id = item["id"]["videoId"]
                published_at_utc = datetime.fromisoformat(item["snippet"]["publishedAt"].replace('Z', '+00:00'))
                
                video_list.append({
                    "video_id": video_id,
                    "title": item["snippet"]["title"],
                    "channelTitle": item["snippet"]["channelTitle"],
                    "published_at_kst": to_iso_kst(published_at_utc.astimezone(KST)),
                    "commentCount": 0 
                })
            
            if video_list:
                df_videos = pd.DataFrame(video_list)
                return df_videos['video_id'].tolist(), df_videos
            
        except HttpError as e:
            if 'quotaExceeded' in str(e) or 'keyInvalid' in str(e):
                st.warning(f"YouTube API 키 할당량 초과 또는 무효: {key[:10]}...")
                continue
            st.error(f"비디오 검색 오류: {e}")
            break
        except Exception as e:
            st.error(f"비디오 검색 중 예기치 않은 오류: {e}")
            break
    return [], None

def fetch_single_video_comments(video_id: str, max_results: int, api_keys: list, result_queue: list):
    """단일 비디오 댓글을 수집하고 결과를 큐에 추가합니다. (Thread worker)"""
    
    total_comments = 0
    next_page_token = None
    
    try:
        # API 키 순환 및 YouTube 서비스 획득 (키 오류시 재시도 3회)
        for attempt in range(len(api_keys)):
            key = api_keys[(attempt + 1) % len(api_keys)] 
            youtube = _get_youtube_service(key)
            if not youtube: continue
            
            comments_data = []
            
            # 수집 루프 (페이지 최대 100개 * 50 = 5000개 제한)
            for _ in range(100): 
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=50,
                    pageToken=next_page_token,
                    order="time"
                )
                response = request.execute()
                
                # 댓글 쓰기
                for item in response.get("items", []):
                    top_level_comment = item["snippet"]["topLevelComment"]["snippet"]
                    
                    comments_data.append({
                        "comment_id": item["snippet"]["topLevelComment"]["id"],
                        "author": top_level_comment["authorDisplayName"],
                        "published_at": top_level_comment["publishedAt"],
                        "text": top_level_comment["textDisplay"].replace('\n', ' ').replace('\r', ' '), 
                        "like_count": top_level_comment["likeCount"],
                        "video_id": video_id
                    })
                    total_comments += 1

                next_page_token = response.get('nextPageToken')
                if not next_page_token or total_comments >= max_results:
                    break
            
            result_queue.append(pd.DataFrame(comments_data))
            return 
            
        # 모든 API 키 시도가 실패한 경우
        result_queue.append(None)
    
    except HttpError as e:
        if 'commentsDisabled' in str(e):
            result_queue.append(f"Comments Disabled:{video_id}")
        elif 'quotaExceeded' in str(e) or 'keyInvalid' in str(e):
             result_queue.append(f"Quota Error:{video_id}")
        else:
             result_queue.append(f"HTTP Error:{video_id}:{e}")
    except Exception:
        result_queue.append(f"Unknown Error:{video_id}")

def concurrent_fetch_comments(video_ids: list, total_comments_limit=5000) -> (pd.DataFrame or None, str):
    """여러 영상 ID에 대해 동시에 댓글을 수집하고 하나의 DataFrame으로 결합합니다."""
    
    if not video_ids: return None, "수집할 영상 ID가 없습니다."

    api_keys = YT_API_KEYS[:]
    result_queue = []
    
    with ThreadPoolExecutor(max_workers=min(len(video_ids), 5)) as executor:
        futures = [executor.submit(fetch_single_video_comments, vid, total_comments_limit // len(video_ids), api_keys, result_queue) for vid in video_ids]
        
        for future in as_completed(futures):
            try:
                future.result() 
            except Exception as e:
                print(f"댓글 수집 스레드에서 오류 발생: {e}")

    successful_dfs = [df for df in result_queue if isinstance(df, pd.DataFrame) and not df.empty]
    error_messages = [msg for msg in result_queue if isinstance(msg, str)]

    if not successful_dfs:
        return None, "모든 영상의 댓글 수집에 실패했거나 댓글이 비활성화되어 있습니다."
    
    df_combined = pd.concat(successful_dfs, ignore_index=True)
    
    filename = os.path.join(BASE_DIR, f"comments_combined_{now_kst().strftime('%Y%m%d%H%M%S')}.csv")
    df_combined.to_csv(filename, index=False, encoding='utf-8-sig')

    return df_combined, filename

# ============== 3) Gemini 로직 (원본 스키마/프롬프트 복원) ==============

def _rotate_gemini_key():
    """Gemini API 키 순환."""
    return _rotate_key(GEMINI_API_KEYS)

def parse_user_query_to_schema(user_query: str, last_video_ids: list) -> dict:
    """
    사용자 질문을 분석하여 유튜브 분석에 필요한 JSON 스키마를 추출합니다.
    (원본 파일의 복잡한 스키마 구조 그대로 유지)
    """
    key = _rotate_gemini_key()
    if not key:
        return {"error": "Gemini API 키가 없습니다."}

    # === 요청대로 원본 파일의 복잡한 스키마 구조 그대로 복원 ===
    schema = {
        "type": "OBJECT",
        "properties": {
            "search_term": {
                "type": "STRING",
                "description": "사용자가 검색을 요청한 경우의 키워드 (예: '신제품 리뷰', '갤럭시 S24'). URL이 아닌 일반 키워드입니다. URL이 포함된 경우 빈 문자열을 반환합니다."
            },
            "video_id": {
                "type": "STRING",
                "description": "사용자가 언급한 단일 유튜브 영상 ID (URL에서 추출된 ID). 단일 영상 분석 시 사용됩니다. 찾지 못했으면 빈 문자열을 반환합니다."
            },
            "analysis_type": {
                "type": "STRING",
                "description": "'FULL_ANALYSIS' (새로운 검색/기간 요청), 'CHAT_FOLLOW_UP' (기존 데이터 기반 추가 질문), 'SIMPLE_QA' (일반 질문) 중 하나."
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
                "description": "댓글 데이터 내에서 특별히 필터링/요약하려는 키워드 리스트 (예: '가격', '성능', '버그')."
            },
            "filter_type": {
                "type": "STRING",
                "description": "댓글 필터링 기준 ('ALL', 'RELEVANCE', 'LIKES' 중 하나). 기본값은 'ALL'."
            },
            "entities": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "댓글 데이터 내에서 언급 빈도/정서 분석에 사용할 명사/엔티티 리스트 (예: '가격', '배터리 수명')."
            }
        },
        "required": ["analysis_type"]
    }
    # ==============================================================
    
    # 기본값 설정: 지난 30일
    today = now_kst()
    a_month_ago = today - timedelta(days=30)
    
    current_schema = {
        "search_term": "",
        "video_id": parse_youtube_url(user_query) if ('youtube.com' in user_query or 'youtu.be' in user_query) else "",
        "start_iso": to_iso_kst(a_month_ago.replace(hour=0, minute=0, second=0, microsecond=0)),
        "end_iso": to_iso_kst(today.replace(hour=23, minute=59, second=59, microsecond=0)),
        "keywords": [],
        "filter_type": "ALL",
        "entities": []
    }
    
    # === 요청대로 원본 파일의 시스템 프롬프트 구조 그대로 복원 ===
    last_ids_str = ', '.join(last_video_ids) if last_video_ids else 'N/A'

    system_prompt = (
        "당신은 사용자 요청을 유튜브 댓글 분석 시스템이 이해할 수 있는 JSON 명령어로 변환하는 AI입니다. "
        "사용자의 질문과 대화 맥락을 고려하여, URL이 있으면 'video_id'에, URL이 없으면 일반 키워드를 'search_term'에 추출하세요. "
        f"마지막으로 분석한 영상 ID들(복수 가능)은 '{last_ids_str}'입니다. 현재 날짜는 {today.strftime('%Y년 %m월 %d일')}입니다. "
        "새로운 영상/검색 요청이면 'FULL_ANALYSIS'로, 기존 데이터 기반 추가 질문이면 'CHAT_FOLLOW_UP', 일반 질문은 'SIMPLE_QA'로 설정합니다. "
        "기간 정보가 없으면 기본값으로 지난 30일을 사용합니다. 'filter_type'이 언급되지 않으면 기본값 'ALL'을 사용합니다. 반드시 JSON 형식으로만 응답해야 합니다."
    )
    # ==============================================================
    
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
        
        for attempt in range(3):
            response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            if response.status_code == 200:
                json_string = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(json_string)
                
                # 빈 문자열을 None으로 간주하고 기존 current_schema의 기본값으로 대체하지 않도록 처리
                final_schema = {**current_schema, **parsed}
                return final_schema

            elif response.status_code == 429 and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            else:
                return {"error": f"API 호출 실패: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def synthesize_response(query: str, analysis_data: str, video_info_df: pd.DataFrame or None, schema: dict) -> str:
    """분석 데이터와 원래 질문을 바탕으로 최종 답변을 생성합니다. (AI 요약)"""
    key = _rotate_gemini_key()
    if not key: return "Gemini API 키가 없습니다."

    system_prompt = (
        "당신은 유튜브 댓글 분석 전문가이자 친절한 챗봇입니다. "
        "제공된 '분석 결과 데이터'와 '비디오 정보(DF)'를 바탕으로 사용자의 질문에 통찰력 있고 이해하기 쉬운 한국어 답변을 제공해야 합니다. "
        "데이터를 그대로 나열하지 않고, 핵심 요약과 통계를 포함하여 자연스러운 문장으로 구성하세요. "
        "만약 '분석 결과 데이터'가 비어 있다면, 비디오 정보만 사용하여 일반적인 답변을 제공하세요."
    )
    
    # 비디오 정보 요약 (여러 개의 비디오를 처리할 수 있도록 문자열로 변환)
    if video_info_df is not None and not video_info_df.empty:
        video_info_str = "분석 대상 영상 목록:\n"
        for _, row in video_info_df.iterrows():
            # video_id를 포함하여 분석된 영상 목록 정보 제공
            video_info_str += f"- 제목: {row['title']} (채널: {row['channelTitle']}) [ID: {row['video_id']}]\n"
    else:
        video_info_str = "N/A"
    
    user_prompt = (
        f"사용자의 질문: '{query}'\n"
        f"분석 요청 스키마: {json.dumps(schema, ensure_ascii=False)}\n"
        f"비디오 기본 정보:\n{video_info_str}\n"
        f"댓글 분석 결과 데이터:\n{analysis_data}\n\n"
        "이 정보를 기반으로 사용자 질문에 답변해주세요. 댓글 데이터를 **요약**하고, 키워드나 기간이 지정되었다면 그 결과를 **핵심 문장**으로 강조하세요. 답변 시, 분석된 영상의 **제목**과 **수**를 언급하여 결과를 구체화해주세요."
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

# ============== 4) 데이터 필터링 및 요약 로직 ==============

def get_filtered_comments_df(comments_file: str, schema: dict) -> pd.DataFrame:
    """댓글 CSV 파일을 읽고, 기간 및 키워드에 따라 필터링합니다."""
    if not comments_file or not os.path.exists(comments_file):
        return None
    
    df_list = []
    try:
        dtype_map = {'comment_id': str, 'author': str, 'published_at': str, 'text': str, 'like_count': 'Int64', 'video_id': str}
        # 메모리 효율을 위해 chunksize 사용
        for chunk in pd.read_csv(comments_file, chunksize=100000, encoding='utf-8-sig', dtype=dtype_map):
            df_list.append(chunk)
        df = pd.concat(df_list, ignore_index=True)
    except Exception as e:
        st.error(f"CSV 파일 로드 오류: {e}")
        return None
    
    if df.empty: return df

    # 1. 기간 필터링
    try:
        start_dt = datetime.fromisoformat(schema['start_iso'])
        end_dt = datetime.fromisoformat(schema['end_iso'])
    except ValueError:
        start_dt = now_kst() - timedelta(days=30)
        end_dt = now_kst()
    
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
    df['published_at_kst'] = df['published_at'].dt.tz_convert(KST)
    
    df_filtered = df[(df['published_at_kst'] >= start_dt) & (df['published_at_kst'] <= end_dt)].copy()
    
    # 2. 키워드 필터링 (댓글 내용 기준)
    keywords = [k for k in schema.get('keywords', []) if k]
    if keywords:
        pattern = '|'.join(re.escape(k) for k in keywords)
        df_filtered = df_filtered[df_filtered['text'].fillna('').str.contains(pattern, case=False, na=False)].copy()

    # 'filter_type'은 정량 분석에 쓰이는 필터였으므로, 현재는 데이터를 필터링하는 용도로만 남기고 다른 로직은 제거
    filter_type = schema.get('filter_type', 'ALL')
    if filter_type == 'LIKES':
         # 좋아요순으로 정렬 후 상위 5000개만 필터링 (최대 댓글 수 초과 방지 및 관련성 높은 댓글 선택)
         df_filtered = df_filtered.sort_values(by='like_count', ascending=False).head(5000).reset_index(drop=True)

    return df_filtered.reset_index(drop=True)

def generate_analysis_summary(df: pd.DataFrame, schema: dict) -> str:
    """필터링된 DataFrame을 기반으로 AI에게 전달할 최소한의 통계 요약 및 댓글 샘플을 생성합니다."""
    # 정량 분석 코드(형태소 분석, 키워드 빈도 등)는 모두 제거하고, 오직 AI 요약에 필요한 원시 데이터만 제공

    if df is None or df.empty:
        return "필터링된 댓글 데이터가 없어 분석을 수행할 수 없습니다. (기간 또는 키워드 불일치)"
    
    summary = f"**[필터링 통계 요약]**\n"
    summary += f"- 최종 분석 대상 댓글 수: {len(df):,}개\n"
    summary += f"- 고유 작성자 수: {df['author'].nunique():,}명\n"
    
    # 최다 좋아요 댓글
    top_liked = df.sort_values(by='like_count', ascending=False).head(1)
    if not top_liked.empty:
        summary += f"- 최다 좋아요 댓글 ({top_liked['like_count'].iloc[0]}개, 작성자: {top_liked['author'].iloc[0]}): \"{top_liked['text'].iloc[0][:150].replace('\n', ' ')}...\"\n"

    # 댓글 샘플 (AI 답변 맥락 유지를 위해 랜덤 20개 선택)
    if len(df) > 20:
        sample_df = df.sample(n=20)
    else:
        sample_df = df
        
    sample_comments = sample_df['text'].str.cat(sep='\n---END---\n').replace('\n', ' ')
    summary += "\n**[댓글 샘플 (총 20개)]**\n"
    summary += sample_comments
    
    return summary

# ============== 5) 메인 챗봇 루프 ==============

# --- Chat Display ---
chat_container = st.container(height=600, border=False) # 채팅 영역을 고정 높이 컨테이너로 설정

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Chat Input Handler ---
if prompt := st.chat_input("YouTube URL 또는 분석 질문을 입력하세요."):
    
    # 1. 사용자 질문 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 컨테이너를 다시 렌더링하여 새 메시지를 표시
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    with st.chat_message("assistant"):
        
        # 2. Gemini 파싱 (질문 -> Schema 명령어)
        with st.spinner("AI가 질문을 이해하고 분석 스키마를 추출하는 중..."):
            parsed_schema = parse_user_query_to_schema(prompt, st.session_state.last_video_ids)
            st.session_state.last_schema = parsed_schema # 스키마 저장 (사이드바 출력용)

        if parsed_schema.get("error"):
            response = f"죄송합니다. 질문 분석 중 오류가 발생했습니다: {parsed_schema['error']}"
            st.error(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            safe_rerun()
            
        video_id = parsed_schema.get("video_id", "")
        search_term = parsed_schema.get("search_term", "")
        analysis_type = parsed_schema.get("analysis_type", "SIMPLE_QA")
        
        df_filtered = None
        df_videos = st.session_state.last_video_info_df # 기존 정보 사용 시도
        final_response = "분석을 시작합니다." # 초기화

        if analysis_type == "SIMPLE_QA":
            # 일반 QA만 요청받은 경우 (데이터 수집 불필요)
            response = synthesize_response(prompt, "N/A", df_videos, parsed_schema)
            st.markdown(response)
            final_response = response
        
        elif analysis_type == "FULL_ANALYSIS":
            
            video_ids_to_fetch = []
            
            # --- 3-1. 비디오 ID(들) 확정 ---
            if video_id:
                # 단일 URL이 입력된 경우
                video_ids_to_fetch = [video_id]
                with st.spinner(f"단일 영상 ID({video_id}) 정보 로드 중..."):
                    single_video_info = search_videos(f"video id {video_id}", max_videos=1)[1] 
                    if single_video_info is not None:
                        df_videos = single_video_info
                    else:
                        response = "단일 비디오 정보를 로드할 수 없습니다. ID 또는 API 키를 확인해주세요."
                        st.error(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        safe_rerun()

            elif search_term:
                # 검색 키워드가 입력된 경우
                start_dt = datetime.fromisoformat(parsed_schema['start_iso'])
                with st.spinner(f"키워드 '{search_term}'로 영상 검색 및 정보 로드 중..."):
                    video_ids_to_fetch, df_videos = search_videos(search_term, published_after=start_dt)
                    if not video_ids_to_fetch:
                        response = f"키워드 '{search_term}'에 대한 영상을 찾을 수 없습니다."
                        st.warning(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        safe_rerun()
            else:
                response = "분석할 영상 URL 또는 검색할 키워드가 없습니다. 다시 질문해주세요."
                st.warning(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                safe_rerun()

            # --- 3-2. 댓글 수집 및 상태 저장 ---
            if video_ids_to_fetch:
                with st.spinner(f"{len(video_ids_to_fetch)}개 영상의 댓글을 수집 중입니다. (약 {len(video_ids_to_fetch)*500}개 목표)..."):
                    df_combined, comments_file = concurrent_fetch_comments(video_ids_to_fetch)

                    if df_combined is None:
                        response = f"댓글 수집에 실패했습니다. API 할당량 또는 권한을 확인해주세요. 오류: {comments_file}"
                        st.error(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        safe_rerun()
                    
                    # 상태 업데이트
                    st.session_state.last_video_ids = video_ids_to_fetch
                    st.session_state.last_comments_file = comments_file
                    st.session_state.last_video_info_df = df_videos # <-- 다중/단일 영상 DF 모두 저장

                # --- 3-3. 데이터 필터링 및 요약 데이터 생성 ---
                with st.spinner("수집된 데이터를 필터링하고 AI 요약을 위한 데이터를 생성하는 중..."):
                    df_filtered = get_filtered_comments_df(comments_file, parsed_schema)
                    analysis_data = generate_analysis_summary(df_filtered, parsed_schema)
                
                # --- 3-4. AI 답변 합성 ---
                with st.spinner("AI가 분석을 기반으로 답변을 생성하는 중..."):
                    final_response = synthesize_response(prompt, analysis_data, df_videos, parsed_schema)
                    st.markdown(final_response)

        
        elif analysis_type == "CHAT_FOLLOW_UP":
            
            # --- 3-1. 기존 데이터 재활용 및 필터링 ---
            if not st.session_state.last_comments_file or not os.path.exists(st.session_state.last_comments_file):
                response = "이전에 분석된 영상 데이터가 없습니다. 키워드 또는 URL과 함께 분석 요청을 먼저 해주세요."
                st.warning(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                safe_rerun()
                
            comments_file = st.session_state.last_comments_file
            
            with st.spinner("기존 데이터를 필터링하고 추가 분석하는 중..."):
                df_filtered = get_filtered_comments_df(comments_file, parsed_schema)
                analysis_data = generate_analysis_summary(df_filtered, parsed_schema)

            # --- 3-2. AI 답변 합성 ---
            with st.spinner("AI가 추가 분석을 기반으로 답변을 생성하는 중..."):
                final_response = synthesize_response(prompt, analysis_data, df_videos, parsed_schema)
                st.markdown(final_response)
        
    st.session_state.messages.append({"role": "assistant", "content": final_response})
    safe_rerun()

# ============== 6) 사이드바 렌더링 (UI/다운로드) ==============

with st.sidebar:
    st.header("⚙️ 분석 상태 및 도구")
    
    if st.session_state.last_video_info_df is not None and not st.session_state.last_video_info_df.empty:
        df = st.session_state.last_video_info_df
        st.subheader(f"마지막 분석 영상 ({len(df)}개)")
        # 5개까지만 보여줌
        info_list = "\n".join([f"- **{row['title']}** (채널: {row['channelTitle']})" for _, row in df.head(5).iterrows()])
        st.info(info_list)
        if len(df) > 5:
             st.caption(f"외 {len(df) - 5}개 영상")
    else:
        st.subheader("분석 영상 대기 중")
        
    st.markdown("---")
    
    st.subheader("적용 필터 정보")
    if st.session_state.last_schema:
        s = st.session_state.last_schema
        st.caption(f"검색어: `{s.get('search_term') or 'N/A'}` / ID: `{s.get('video_id') or 'N/A'}`")
        st.markdown(f"- **기간:** `{s.get('start_iso', 'N/A').split('T')[0]} ~ {s.get('end_iso', 'N/A').split('T')[0]}`")
        keywords = s.get('keywords', [])
        entities = s.get('entities', [])
        st.markdown(f"- **댓글 키워드:** {', '.join(keywords) or '(전체)'}")
        st.markdown(f"- **분석 엔티티:** {', '.join(entities) or '(없음)'}")
        st.markdown(f"- **필터 유형:** `{s.get('filter_type', 'ALL')}`")
        
    st.markdown("---")
    
    st.subheader("다운로드")
    # 전체 댓글 CSV 다운로드
    if st.session_state.last_comments_file and os.path.exists(st.session_state.last_comments_file):
        try:
            with open(st.session_state.last_comments_file, "rb") as f:
                st.download_button(
                    "⬇️ 수집된 댓글 CSV 다운로드",
                    data=f.read(),
                    file_name=f"comments_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        except Exception:
             st.caption("댓글 파일 로드 실패")
    else:
        st.caption("분석된 댓글 파일 없음")

    # 영상 목록 CSV 다운로드
    if st.session_state.last_video_info_df is not None and not st.session_state.last_video_info_df.empty:
        csv_videos = st.session_state.last_video_info_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ 분석 영상 목록 CSV 다운로드",
            data=csv_videos,
            file_name=f"videos_{len(st.session_state.last_video_info_df)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.caption("분석 영상 목록 파일 없음")
        
    st.markdown("---")
    
    # 초기화 버튼
    if st.button("🔄 세션 초기화", type="secondary"):
        st.session_state.clear()
        safe_rerun()
