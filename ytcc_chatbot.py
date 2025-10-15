# -*- coding: utf-8 -*-
# 💬 유튜브 댓글분석기 — 순수 챗봇 모드 (세션 관리 기능 강화)

import streamlit as st
import pandas as pd
import os, re, gc, time, json, base64, requests
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4
import io

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai
from streamlit.components.v1 import html as st_html

# -------------------- 페이지/전역 --------------------
st.set_page_config(page_title="유튜브 댓글분석: 챗봇", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* Streamlit 메인 컨테이너 패딩 최소화 */
.main .block-container {
    padding-top: 2rem;
    padding-right: 1rem;
    padding-left: 1rem;
    padding-bottom: 5rem;
}
[data-testid="stSidebarContent"] {
    padding-top: 1.5rem;
}
header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* --- [추가] 사이드바 너비 고정 --- */
[data-testid="stSidebar"] {
    width: 350px !important;
    min-width: 350px !important;
    max-width: 350px !important;
}
/* --- [추가] 사이드바 리사이즈 핸들 숨기기 --- */
[data-testid="stSidebar"] + div[class*="resizer"] {
    display: none;
}
/* --- --- */

/* AI 답변 폰트 크기 조정 */
[data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) p,
[data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) li {
    font-size: 0.95rem;
}
/* 다운로드 버튼을 텍스트 링크처럼 보이게 스타일링 */
.stDownloadButton button {
    background-color: transparent;
    color: #1c83e1;
    border: none;
    padding: 0;
    text-decoration: underline;
    font-size: 14px;
    font-weight: normal;
}
.stDownloadButton button:hover {
    color: #0b5cab;
}
</style>
""", unsafe_allow_html=True)

# --- 경로 및 GitHub 설정 ---
BASE_DIR = "/tmp"
SESS_DIR = os.path.join(BASE_DIR, "sessions")
os.makedirs(SESS_DIR, exist_ok=True)

GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "")
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")

KST = timezone(timedelta(hours=9))
def now_kst(): return datetime.now(tz=KST)
def to_iso_kst(dt: datetime) -> str:
    if dt.tzinfo is None: dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST).isoformat(timespec="seconds")
def kst_to_rfc3339_utc(dt_kst: datetime) -> str:
    if dt_kst.tzinfo is None: dt_kst = dt_kst.replace(tzinfo=KST)
    return dt_kst.astimezone(timezone.utc).isoformat().replace("+00:00","Z")

# -------------------- 키/상수 --------------------
_YT_FALLBACK, _GEM_FALLBACK = [], []
YT_API_KEYS       = list(st.secrets.get("YT_API_KEYS", [])) or _YT_FALLBACK
GEMINI_API_KEYS   = list(st.secrets.get("GEMINI_API_KEYS", [])) or _GEM_FALLBACK
GEMINI_MODEL      = "gemini-2.0-flash-lite"
GEMINI_TIMEOUT    = 120
GEMINI_MAX_TOKENS = 2048
MAX_TOTAL_COMMENTS   = 120_000
MAX_COMMENTS_PER_VID = 4_000

# -------------------- 상태 --------------------
def ensure_state():
    defaults = {"chat":[], "last_schema":None, "last_csv":"", "last_df":None, "sample_text":"", "loaded_session_name": None}
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
ensure_state()

# --- GitHub API 함수 ---
def _gh_headers(token: str):
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

def github_upload_file(repo, branch, path_in_repo, local_path, token):
    url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
    with open(local_path, "rb") as f: content = base64.b64encode(f.read()).decode()
    headers = _gh_headers(token)
    get_resp = requests.get(url + f"?ref={branch}", headers=headers)
    sha = get_resp.json().get("sha") if get_resp.ok else None
    data = {"message": f"archive: {os.path.basename(path_in_repo)}", "content": content, "branch": branch}
    if sha: data["sha"] = sha
    resp = requests.put(url, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()

def github_list_dir(repo, branch, folder, token):
    url = f"https://api.github.com/repos/{repo}/contents/{folder}?ref={branch}"
    resp = requests.get(url, headers=_gh_headers(token))
    return [item['name'] for item in resp.json() if item['type'] == 'dir'] if resp.ok else []

def github_download_file(repo, branch, path_in_repo, token, local_path):
    url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}?ref={branch}"
    resp = requests.get(url, headers=_gh_headers(token))
    if resp.ok:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f: f.write(base64.b64decode(resp.json()["content"]))
        return True
    return False

# [추가] GitHub 파일/폴더 삭제 함수
def github_delete_folder(repo, branch, folder_path, token):
    # 1. 폴더 내용물 리스트업
    contents_url = f"https://api.github.com/repos/{repo}/contents/{folder_path}?ref={branch}"
    headers = _gh_headers(token)
    resp = requests.get(contents_url, headers=headers)
    if not resp.ok: return
    
    # 2. 각 파일 삭제
    for item in resp.json():
        delete_url = f"https://api.github.com/repos/{repo}/contents/{item['path']}"
        data = {
            "message": f"delete: {item['name']}",
            "sha": item['sha'],
            "branch": branch
        }
        requests.delete(delete_url, headers=headers, json=data)

# --- 세션 관리 함수 ---
# [수정] 세션 이름 규칙 변경 및 덮어쓰기 로직
def _build_session_name() -> str:
    # 이미 불러온 세션이 있다면 그 이름을 그대로 사용 (덮어쓰기 위함)
    if st.session_state.get("loaded_session_name"):
        return st.session_state.loaded_session_name

    schema = st.session_state.get("last_schema", {})
    kw = (schema.get("keywords", ["NoKeyword"]))[0]
    kw_slug = re.sub(r'[^\w-]', '', kw.replace(' ', '_'))[:20]
    # 이름 규칙 변경: 주제키워드_세션생성시점
    return f"{kw_slug}_{now_kst().strftime('%Y-%m-%d_%H%M')}"

def save_current_session_to_github():
    if not all([GITHUB_REPO, GITHUB_TOKEN, st.session_state.chat, st.session_state.last_csv]):
        st.sidebar.warning("저장할 데이터가 없거나 GitHub 설정이 누락되었습니다.")
        return False, ""
    
    sess_name = _build_session_name()
    local_dir = os.path.join(SESS_DIR, sess_name)
    os.makedirs(local_dir, exist_ok=True)

    try:
        meta_path = os.path.join(local_dir, "qa.json")
        meta_data = {"chat": st.session_state.chat, "last_schema": st.session_state.last_schema}
        with open(meta_path, "w", encoding="utf-8") as f: json.dump(meta_data, f, ensure_ascii=False, indent=2)

        comments_path = os.path.join(local_dir, "comments.csv")
        videos_path = os.path.join(local_dir, "videos.csv")
        os.system(f'cp "{st.session_state.last_csv}" "{comments_path}"')
        if st.session_state.last_df is not None:
            st.session_state.last_df.to_csv(videos_path, index=False, encoding="utf-8-sig")

        github_upload_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/qa.json", meta_path, GITHUB_TOKEN)
        github_upload_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/comments.csv", comments_path, GITHUB_TOKEN)
        if os.path.exists(videos_path):
            github_upload_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/videos.csv", videos_path, GITHUB_TOKEN)
        return True, sess_name
    except Exception as e:
        st.sidebar.error(f"저장 실패: {e}")
        return False, ""

def load_session_from_github(sess_name: str):
    with st.spinner(f"세션 '{sess_name}' 불러오는 중..."):
        try:
            local_dir = os.path.join(SESS_DIR, sess_name)
            qa_ok = github_download_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/qa.json", GITHUB_TOKEN, os.path.join(local_dir, "qa.json"))
            comments_ok = github_download_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/comments.csv", GITHUB_TOKEN, os.path.join(local_dir, "comments.csv"))
            videos_ok = github_download_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/videos.csv", GITHUB_TOKEN, os.path.join(local_dir, "videos.csv"))

            if not (qa_ok and comments_ok):
                st.error("세션 핵심 파일(qa.json, comments.csv)을 불러오는 데 실패했습니다."); return

            st.session_state.clear(); ensure_state()
            
            with open(os.path.join(local_dir, "qa.json"), "r", encoding="utf-8") as f: meta = json.load(f)
            
            st.session_state.update({
                "chat": meta.get("chat", []), "last_schema": meta.get("last_schema", None),
                "last_csv": os.path.join(local_dir, "comments.csv"),
                "last_df": pd.read_csv(os.path.join(local_dir, "videos.csv")) if videos_ok and os.path.exists(os.path.join(local_dir, "videos.csv")) else pd.DataFrame(),
                "loaded_session_name": sess_name, # [추가] 불러온 세션 이름 저장
            })
            st.session_state.sample_text, _, _ = serialize_comments_for_llm_from_file(st.session_state.last_csv)
        except Exception as e:
            st.error(f"세션 로드 실패: {e}")

# --- 최상단에서 세션 로드/삭제 요청 처리 ---
if 'session_to_load' in st.session_state:
    sess_name = st.session_state.pop('session_to_load')
    load_session_from_github(sess_name)
    st.rerun()

if 'session_to_delete' in st.session_state:
    sess_name = st.session_state.pop('session_to_delete')
    with st.spinner(f"세션 '{sess_name}' 삭제 중..."):
        github_delete_folder(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}", GITHUB_TOKEN)
    st.success(f"세션 '{sess_name}' 삭제 완료.")
    time.sleep(1)
    st.rerun()

# -------------------- 사이드바 (레이아웃 수정) --------------------
with st.sidebar:
    st.markdown("""
    <style>
        [data-testid="stSidebarUserContent"] { display: flex; flex-direction: column; height: calc(100vh - 4rem); }
        .sidebar-top-section { flex-grow: 1; overflow-y: auto; }
        .sidebar-bottom-section { flex-shrink: 0; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-top-section">', unsafe_allow_html=True)
    if st.button("✨ 새 채팅", use_container_width=True, type="secondary"):
        st.session_state.clear(); st.rerun()

    if st.session_state.chat and st.session_state.last_csv:
        if st.button("💾 현재 대화 저장", use_container_width=True):
            with st.spinner("세션 저장 중..."): success, sess_name = save_current_session_to_github()
            if success: st.success(f"'{sess_name}' 저장 완료!"); time.sleep(2); st.rerun()

    st.markdown("---"); st.markdown("#### 대화 기록")

    if not all([GITHUB_TOKEN, GITHUB_REPO]):
        st.caption("GitHub 설정이 Secrets에 없습니다.")
    else:
        try:
            sessions = sorted(github_list_dir(GITHUB_REPO, GITHUB_BRANCH, "sessions", GITHUB_TOKEN), reverse=True)
            if not sessions:
                st.caption("저장된 기록이 없습니다.")
            else:
                for sess in sessions:
                    col1, col2 = st.columns([0.85, 0.15])
                    if col1.button(sess, key=f"sess_{sess}", use_container_width=True):
                        st.session_state.session_to_load = sess; st.rerun()
                    if col2.button("🗑️", key=f"del_{sess}", use_container_width=True):
                        st.session_state.session_to_delete = sess; st.rerun()
        except Exception as e:
            st.error("기록 로딩 실패")
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-bottom-section">', unsafe_allow_html=True)
    st.markdown("""
        <hr><h3>📞 문의</h3><p>미디어)디지털마케팅 데이터파트 김호범</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- 로직 (이하 수정 없음) --------------------
def scroll_to_bottom(): # ... (이하 모든 함수는 이전과 동일)
    st_html("<script> let last_message = document.querySelectorAll('.stChatMessage'); if (last_message.length > 0) { last_message[last_message.length - 1].scrollIntoView({behavior: 'smooth'}); } </script>", height=0)

def render_metadata_and_downloads():
    if not st.session_state.get("last_schema"): return
    schema = st.session_state["last_schema"]
    kw_main = schema.get("keywords", [])
    start_iso, end_iso = schema.get('start_iso', ''), schema.get('end_iso', '')
    try:
        start_dt_str = datetime.fromisoformat(start_iso).astimezone(KST).strftime('%Y-%m-%d %H:%M')
        end_dt_str = datetime.fromisoformat(end_iso).astimezone(KST).strftime('%Y-%m-%d %H:%M')
    except (ValueError, TypeError):
        start_dt_str = start_iso.split('T')[0] if start_iso else ""
        end_dt_str = end_iso.split('T')[0] if end_iso else ""
    with st.container(border=True):
        st.markdown(f"""<div style="font-size:14px; color:#4b5563; line-height: 1.8;"><span style='font-weight:600;'>키워드:</span> {', '.join(kw_main) if kw_main else '(없음)'}<br><span style='font-weight:600;'>기간:</span> {start_dt_str} ~ {end_dt_str} (KST)</div>""", unsafe_allow_html=True)
        csv_path, df_videos = st.session_state.get("last_csv"), st.session_state.get("last_df")
        if csv_path and os.path.exists(csv_path) and df_videos is not None and not df_videos.empty:
            with open(csv_path, "rb") as f: comment_csv_data = f.read()
            buffer = io.BytesIO()
            df_videos.to_csv(buffer, index=False, encoding="utf-8-sig")
            video_csv_data = buffer.getvalue()
            keywords_str = "_".join(kw_main).replace(" ", "_") if kw_main else "data"
            now_str = now_kst().strftime('%Y%m%d')
            col1, col2, col3, _ = st.columns([1.1, 1.2, 1.2, 6.5])
            col1.markdown("<div style='font-size:14px; color:#4b5563; font-weight:600; padding-top: 5px;'>다운로드:</div>", unsafe_allow_html=True)
            with col2: st.download_button("전체댓글", comment_csv_data, f"comments_{keywords_str}_{now_str}.csv", "text/csv")
            with col3: st.download_button("영상목록", video_csv_data, f"videos_{keywords_str}_{now_str}.csv", "text/csv")

def render_chat():
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

class RotatingKeys:
    def __init__(self, keys, state_key: str, on_rotate=None):
        self.keys, self.state_key, self.on_rotate = [k.strip() for k in (keys or []) if isinstance(k, str) and k.strip()][:10], state_key, on_rotate
        idx = st.session_state.get(state_key, 0)
        self.idx = 0 if not self.keys else (idx % len(self.keys))
        st.session_state[state_key] = self.idx
    def current(self): return self.keys[self.idx % len(self.keys)] if self.keys else None
    def rotate(self):
        if not self.keys: return
        self.idx = (self.idx + 1) % len(self.keys)
        st.session_state[self.state_key] = self.idx
        if callable(self.on_rotate): self.on_rotate(self.idx, self.current())

class RotatingYouTube:
    def __init__(self, keys, state_key="yt_key_idx"):
        self.rot, self.service = RotatingKeys(keys, state_key), None
        self._build()
    def _build(self):
        if not (key := self.rot.current()): raise RuntimeError("YouTube API Key가 비어 있습니다.")
        self.service = build("youtube", "v3", developerKey=key)
    def execute(self, factory):
        try: return factory(self.service).execute()
        except HttpError as e:
            status, msg = getattr(getattr(e, 'resp', None), 'status', None), (getattr(e, 'content', b'').decode('utf-8', 'ignore') or '').lower()
            if status in (403, 429) and any(t in msg for t in ["quota", "rate", "limit"]) and len(YT_API_KEYS) > 1:
                self.rot.rotate(); self._build()
                return factory(self.service).execute()
            raise

LIGHT_PROMPT = (f"역할: 유튜브 댓글 반응 분석기의 자연어 해석가.\n목표: 한국어 입력에서 [기간(KST)]과 [키워드/엔티티/옵션]을 해석.\n규칙:\n- 기간은 Asia/Seoul 기준, 상대기간의 종료는 지금.\n- 옵션 탐지: include_replies, channel_filter(any|official|unofficial), lang(ko|en|auto).\n\n출력(6줄 고정):\n- 한 줄 요약: <문장>\n- 기간(KST): <YYYY-MM-DDTHH:MM:SS+09:00> ~ <YYYY-MM-DDTHH:MM:SS+09:00>\n- 키워드: [<메인1>, <메인2>…]\n- 엔티티/보조: [<보조들>]\n- 옵션: {{ include_replies: true|false, channel_filter: \"any|official|unofficial\", lang: \"ko|en|auto\" }}\n- 원문: {{USER_QUERY}}\n\n현재 KST: {to_iso_kst(now_kst())}\n입력:\n{{USER_QUERY}}")

def is_gemini_quota_error(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return ("429" in msg) or ("too many requests" in msg) or ("rate limit" in msg) or ("resource exhausted" in msg) or ("quota" in msg)

def call_gemini_rotating(model_name, keys, system_instruction, user_payload, timeout_s=120, max_tokens=2048) -> str:
    rk = RotatingKeys(keys, "gem_key_idx")
    if not rk.current(): raise RuntimeError("Gemini API Key가 비어 있습니다.")
    for _ in range(len(rk.keys) or 1):
        try:
            genai.configure(api_key=rk.current())
            model = genai.GenerativeModel(model_name, generation_config={"temperature": 0.2, "max_output_tokens": max_tokens})
            resp = model.generate_content([system_instruction, user_payload], request_options={"timeout": timeout_s})
            if out := getattr(resp, "text", None): return out
            if c0 := (getattr(resp, "candidates", None) or [None])[0]:
                if p0 := (getattr(c0, "content", None) and getattr(c0.content, "parts", None) or [None])[0]:
                    if hasattr(p0, "text"): return p0.text
            return ""
        except Exception as e:
            if is_gemini_quota_error(e) and len(rk.keys) > 1: rk.rotate(); continue
            raise
    return ""

def parse_light_block_to_schema(light_text: str) -> dict:
    raw = (light_text or "").strip()
    m_time = re.search(r"기간\(KST\)\s*:\s*([^~]+)~\s*([^\n]+)", raw)
    start_iso, end_iso = (m_time.group(1).strip(), m_time.group(2).strip()) if m_time else (None, None)
    m_kw = re.search(r"키워드\s*:\s*\[(.*?)\]", raw, flags=re.DOTALL)
    keywords = [p.strip() for p in re.split(r"\s*,\s*", m_kw.group(1)) if p.strip()] if m_kw else []
    m_ent = re.search(r"엔티티/보조\s*:\s*\[(.*?)\]", raw, flags=re.DOTALL)
    entities = [p.strip() for p in re.split(r"\s*,\s*", m_ent.group(1)) if p.strip()] if m_ent else []
    m_opt = re.search(r"옵션\s*:\s*\{(.*?)\}", raw, flags=re.DOTALL)
    options = {"include_replies": False, "channel_filter": "any", "lang": "auto"}
    if m_opt:
        blob = m_opt.group(1)
        if ir := re.search(r"include_replies\s*:\s*(true|false)", blob, re.I): options["include_replies"] = (ir.group(1).lower() == "true")
        if cf := re.search(r"channel_filter\s*:\s*\"(any|official|unofficial)\"", blob, re.I): options["channel_filter"] = cf.group(1)
        if lg := re.search(r"lang\s*:\s*\"(ko|en|auto)\"", blob, re.I): options["lang"] = lg.group(1)
    if not (start_iso and end_iso):
        end_dt, start_dt = now_kst(), now_kst() - timedelta(hours=24)
        start_iso, end_iso = to_iso_kst(start_dt), to_iso_kst(end_dt)
    if not keywords: keywords = [m[0]] if (m := re.findall(r"[가-힣A-Za-z0-9]{2,}", raw)) else ["유튜브"]
    return {"start_iso": start_iso, "end_iso": end_iso, "keywords": keywords, "entities": entities, "options": options, "raw": raw}

def yt_search_videos(rt, keyword, max_results, order="relevance", published_after=None, published_before=None):
    video_ids, token = [], None
    while len(video_ids) < max_results:
        params = {"q": keyword, "part": "id", "type": "video", "order": order, "maxResults": min(50, max_results - len(video_ids))}
        if published_after: params["publishedAfter"] = published_after
        if published_before: params["publishedBefore"] = published_before
        if token: params["pageToken"] = token
        resp = rt.execute(lambda s: s.search().list(**params))
        video_ids.extend(it["id"]["videoId"] for it in resp.get("items", []) if it["id"]["videoId"] not in video_ids)
        if not (token := resp.get("nextPageToken")): break
        time.sleep(0.25)
    return video_ids

def yt_video_statistics(rt, video_ids):
    rows = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        if not batch: continue
        resp = rt.execute(lambda s: s.videos().list(part="statistics,snippet,contentDetails", id=",".join(batch)))
        for item in resp.get("items", []):
            stats, snip, cont = item.get("statistics", {}), item.get("snippet", {}), item.get("contentDetails", {})
            dur = cont.get("duration", "")
            h, m, s = re.search(r"(\d+)H", dur), re.search(r"(\d+)M", dur), re.search(r"(\d+)S", dur)
            dur_sec = (int(h.group(1)) * 3600 if h else 0) + (int(m.group(1)) * 60 if m else 0) + (int(s.group(1)) if s else 0)
            vid_id = item.get("id")
            rows.append({"video_id": vid_id, "video_url": f"https://www.youtube.com/watch?v={vid_id}", "title": snip.get("title", ""), "channelTitle": snip.get("channelTitle", ""), "publishedAt": snip.get("publishedAt", ""), "duration": dur, "shortType": "Shorts" if dur_sec <= 60 else "Clip", "viewCount": int(stats.get("viewCount", 0) or 0), "likeCount": int(stats.get("likeCount", 0) or 0), "commentCount": int(stats.get("commentCount", 0) or 0)})
        time.sleep(0.25)
    return rows

def yt_all_replies(rt, parent_id, video_id, title="", short_type="Clip", cap=None):
    replies, token = [], None
    while not (cap is not None and len(replies) >= cap):
        try: resp = rt.execute(lambda s: s.comments().list(part="snippet", parentId=parent_id, maxResults=100, pageToken=token, textFormat="plainText"))
        except HttpError: break
        for c in resp.get("items", []):
            sn = c["snippet"]
            replies.append({"video_id": video_id, "video_title": title, "shortType": short_type, "comment_id": c.get("id", ""), "parent_id": parent_id, "isReply": 1, "author": sn.get("authorDisplayName", ""), "text": sn.get("textDisplay", "") or "", "publishedAt": sn.get("publishedAt", ""), "likeCount": int(sn.get("likeCount", 0) or 0)})
        if not (token := resp.get("nextPageToken")): break
        time.sleep(0.2)
    return replies[:cap] if cap is not None else replies

def yt_all_comments_sync(rt, video_id, title="", short_type="Clip", include_replies=True, max_per_video=None):
    rows, token = [], None
    while not (max_per_video is not None and len(rows) >= max_per_video):
        try: resp = rt.execute(lambda s: s.commentThreads().list(part="snippet,replies", videoId=video_id, maxResults=100, pageToken=token, textFormat="plainText"))
        except HttpError: break
        for it in resp.get("items", []):
            top, thread_id = it["snippet"]["topLevelComment"]["snippet"], it["snippet"]["topLevelComment"]["id"]
            rows.append({"video_id": video_id, "video_title": title, "shortType": short_type, "comment_id": thread_id, "parent_id": "", "isReply": 0, "author": top.get("authorDisplayName", ""), "text": top.get("textDisplay", "") or "", "publishedAt": top.get("publishedAt", ""), "likeCount": int(top.get("likeCount", 0) or 0)})
            if include_replies and int(it["snippet"].get("totalReplyCount", 0) or 0) > 0:
                cap = None if max_per_video is None else max(0, max_per_video - len(rows))
                if cap == 0: break
                rows.extend(yt_all_replies(rt, thread_id, video_id, title, short_type, cap=cap))
        if not (token := resp.get("nextPageToken")): break
        time.sleep(0.2)
    return rows[:max_per_video] if max_per_video is not None else rows

def parallel_collect_comments_streaming(video_list, rt_keys, include_replies, max_total_comments, max_per_video, prog_bar):
    out_csv = os.path.join(BASE_DIR, f"collect_{uuid4().hex}.csv")
    wrote_header, total_written, done, total_videos = False, 0, 0, len(video_list)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(yt_all_comments_sync, RotatingYouTube(rt_keys), v["video_id"], v.get("title", ""), v.get("shortType", "Clip"), include_replies, max_per_video): v for v in video_list}
        for f in as_completed(futures):
            try:
                if comm := f.result():
                    dfc = pd.DataFrame(comm)
                    dfc.to_csv(out_csv, index=False, mode="a" if wrote_header else "w", header=not wrote_header, encoding="utf-8-sig")
                    wrote_header, total_written = True, total_written + len(dfc)
            except Exception: pass
            done += 1
            prog_bar.progress(min(0.90, 0.50 + (done / total_videos) * 0.40 if total_videos > 0 else 0.50), text="댓글 수집중…")
            if total_written >= max_total_comments: break
    return out_csv, total_written

def serialize_comments_for_llm_from_file(csv_path: str, max_chars_per_comment=280, max_total_chars=420_000):
    if not os.path.exists(csv_path): return "", 0, 0
    try: df_all = pd.read_csv(csv_path)
    except Exception: return "", 0, 0
    if df_all.empty: return "", 0, 0
    df_top_likes = df_all.sort_values("likeCount", ascending=False).head(1000)
    df_remaining = df_all.drop(df_top_likes.index)
    df_random = df_remaining.sample(n=min(1000, len(df_remaining))) if not df_remaining.empty else pd.DataFrame()
    df_sample, lines, total_chars = pd.concat([df_top_likes, df_random]), [], 0
    for _, r in df_sample.iterrows():
        if total_chars >= max_total_chars: break
        text = str(r.get("text", "") or "").replace("\n", " ")
        line = f"[{'R' if int(r.get('isReply', 0)) == 1 else 'T'}|♥{int(r.get('likeCount', 0))}] {str(r.get('author', '')).replace('\n', ' ')}: {text[:max_chars_per_comment] + '…' if len(text) > max_chars_per_comment else text}"
        if total_chars + len(line) + 1 > max_total_chars: break
        lines.append(line); total_chars += len(line) + 1
    return "\n".join(lines), len(lines), total_chars

TITLE_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s+.*$")
HEADER_DUP_RE = re.compile(r"유튜브\s*댓글\s*분석.*", re.IGNORECASE)

def tidy_answer(md: str) -> str:
    if not md: return md
    lines = [line for line in md.splitlines() if not (TITLE_LINE_RE.match(line) or HEADER_DUP_RE.search(line))]
    cleaned, prev_blank = [], False
    for l in lines:
        is_blank = not l.strip()
        if is_blank and prev_blank: continue
        cleaned.append(l); prev_blank = is_blank
    return "\n".join(cleaned).strip()

def run_pipeline_first_turn(user_query: str):
    prog_bar = st.progress(0, text="준비 중…")
    if not GEMINI_API_KEYS: return "오류: Gemini API Key가 설정되지 않았습니다."
    prog_bar.progress(0.05, text="해석중…")
    light = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, "", LIGHT_PROMPT.replace("{USER_QUERY}", user_query))
    schema = parse_light_block_to_schema(light)
    st.session_state["last_schema"] = schema
    prog_bar.progress(0.10, text="영상 수집중…")
    if not YT_API_KEYS: return "오류: YouTube API Key가 설정되지 않았습니다."
    rt = RotatingYouTube(YT_API_KEYS)
    start_dt, end_dt = datetime.fromisoformat(schema["start_iso"]), datetime.fromisoformat(schema["end_iso"])
    kw_main, kw_ent = schema.get("keywords", []), schema.get("entities", [])
    all_ids = []
    for base_kw in (kw_main or ["유튜브"]):
        all_ids.extend(yt_search_videos(rt, base_kw, 60, "relevance", kst_to_rfc3339_utc(start_dt), kst_to_rfc3339_utc(end_dt)))
        for e in kw_ent: all_ids.extend(yt_search_videos(rt, f"{base_kw} {e}", 30, "relevance", kst_to_rfc3339_utc(start_dt), kst_to_rfc3339_utc(end_dt)))
    all_ids = list(dict.fromkeys(all_ids))
    prog_bar.progress(0.40, text="댓글 수집 준비중…")
    df_stats = pd.DataFrame(yt_video_statistics(rt, all_ids))
    st.session_state["last_df"] = df_stats
    csv_path, total_cnt = parallel_collect_comments_streaming(df_stats.to_dict('records'), YT_API_KEYS, bool(schema.get("options", {}).get("include_replies")), MAX_TOTAL_COMMENTS, MAX_COMMENTS_PER_VID, prog_bar)
    st.session_state["last_csv"] = csv_path
    if total_cnt == 0: return "지정 기간/키워드에서 댓글을 찾을 수 없습니다. 다른 조건으로 시도해 보세요."
    prog_bar.progress(0.90, text="AI 분석중…")
    sample_text, _, _ = serialize_comments_for_llm_from_file(csv_path)
    st.session_state["sample_text"] = sample_text
    sys = "너는 유튜브 댓글을 분석하는 어시스턴트다. 주어진 댓글 샘플을 바탕으로 핵심 포인트를 항목화하고, 긍/부/중 비율과 대표 코멘트(10개 미만)를 제시하라."
    payload = f"[키워드]: {', '.join(kw_main)}\n[엔티티]: {', '.join(kw_ent)}\n[기간(KST)]: {schema['start_iso']} ~ {schema['end_iso']}\n\n[댓글 샘플]:\n{sample_text}\n"
    answer_md_raw = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, sys, payload)
    prog_bar.progress(1.0, text="완료"); time.sleep(0.5); prog_bar.empty()
    return tidy_answer(answer_md_raw)

def run_followup_turn(user_query: str):
    if not (schema := st.session_state.get("last_schema")): return "오류: 이전 분석 기록이 없습니다. 새 채팅을 시작해주세요."
    sample_text = st.session_state.get("sample_text", "")
    context = "\n".join(f"[이전 {'Q' if m['role'] == 'user' else 'A'}]: {m['content']}" for m in st.session_state["chat"][-10:])
    sys = "너는 유튜브 댓글 분석가다. 주어진 댓글 샘플과 이전 대화 맥락을 바탕으로 현재 질문에 간결하게 답하라. 반드시 댓글 샘플을 근거로 답하고, 인용은 5개 이하로 하라."
    payload = f"{context}\n\n[현재 질문]: {user_query}\n[기간(KST)]: {schema.get('start_iso', '?')} ~ {schema.get('end_iso', '?')}\n\n[댓글 샘플]:\n{sample_text}\n"
    with st.spinner("💬 AI가 답변을 구성 중입니다..."):
        response = tidy_answer(call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, sys, payload))
    return response

# -------------------- 메인 화면 및 실행 로직 --------------------
if not st.session_state.chat:
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; height: 70vh;">
            <h1 style="font-size: 3.5rem; font-weight: 600; background: -webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570, #F2A60C); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">유튜브 댓글분석: AI 챗봇</h1>
            <p style="font-size: 1.2rem; color: #4b5563;">기간과 분석주제를 명시하여 대화를 시작하세요</p>
            <div style="margin-top: 3rem; padding: 1rem 1.5rem; border: 1px solid #e5e7eb; border-radius: 12px; background-color: #fafafa; max-width: 600px;">
                <h4 style="margin-bottom: 1rem; font-weight: 600;">⚠️ 사용 주의사항</h4>
                <ol style="text-align: left; padding-left: 20px;">
                    <li><strong>첫 질문 시</strong> 댓글 수집 및 AI 분석에 다소 시간이 소요될 수 있습니다.</li>
                    <li>한 세션에서는 <strong>하나의 주제</strong>와 관련된 질문만 진행해야 분석 정확도가 유지됩니다.</li>
                </ol>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    render_metadata_and_downloads()
    render_chat()
    scroll_to_bottom()

if prompt := st.chat_input("예) 최근 24시간 태풍상사 김준호 반응 요약해줘"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.chat and st.session_state.chat[-1]["role"] == "user":
    user_query = st.session_state.chat[-1]["content"]
    if not st.session_state.get("last_csv"):
        response = run_pipeline_first_turn(user_query)
    else:
        response = run_followup_turn(user_query)
    st.session_state.chat.append({"role": "assistant", "content": response})
    st.rerun()
