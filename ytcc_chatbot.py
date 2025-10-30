# -*- coding: utf-8 -*-
# 💬 유튜브 댓글분석기 — 순수 챗봇 모드 (세션 관리 + URL 수집 + 안정적 시각화)
# - 첫 질문 후 rerun에 의한 차트 깜빡임 해결 (render_viz_after_response 플래그)
# - 키워드 시각화: Treemap (단일 색조/히트맵 스타일)
# - 1 2 / 3 4 레이아웃, 축 라벨 제거, 폰트 다운스케일
# - 후속 질문에서도 안정적 유지

import streamlit as st
import pandas as pd
import os, re, gc, time, json, base64, requests, io
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai
from streamlit.components.v1 import html as st_html

import plotly.express as px
from plotly import graph_objects as go
import numpy as np

# ======================================================================
# 페이지/전역 설정
# ======================================================================
st.set_page_config(
    page_title="유튜브 댓글분석: 챗봇",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
  .main .block-container { padding-top: 2rem; padding-right: 1rem; padding-left: 1rem; padding-bottom: 5rem; }
  [data-testid="stSidebarContent"] { padding-top: 1.5rem; }
  header, footer, #MainMenu {visibility: hidden;}

  /* 사이드바 폭 고정 */
  [data-testid="stSidebar"] { width: 350px !important; min-width: 350px !important; max-width: 350px !important; }
  [data-testid="stSidebar"] + div[class*="resizer"] { display: none; }

  /* AI 답변 폰트 */
  [data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) p,
  [data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) li { font-size: 0.95rem; }

  /* 다운로드 버튼을 텍스트 링크처럼 */
  .stDownloadButton button { background: transparent; color:#1c83e1; border:none; padding:0; text-decoration:underline; font-size:14px; font-weight:normal;}
  .stDownloadButton button:hover { color:#0b5cab; }

  /* 세션 목록 버튼 */
  .session-list .stButton button { font-size:0.9rem; text-align:left; font-weight:normal; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; display:block; }

  .new-chat-btn button { background-color:#e8f0fe; color:#0052CC !important; border:1px solid #d2e3fc !important; }
  .new-chat-btn button:hover { background-color:#d2e3fc; color:#0041A3 !important; border:1px solid #c2d8f8 !important; }

  /* Plotly 간격 축소 */
  .stPlotlyChart { padding: 0.25rem 0 0 0; }
</style>
""",
    unsafe_allow_html=True
)

# === 공통 Plotly 축소 레이아웃 헬퍼 ===
def _small_fig(fig, *, height=260, title_size=14, font_size=12, legend_size=11, margin=(10,10,28,10)):
    l, r, t, b = margin
    fig.update_layout(
        height=height,
        margin=dict(l=l, r=r, t=t, b=b),
        font=dict(size=font_size),
        title_font=dict(size=title_size),
        legend=dict(font=dict(size=legend_size))
    )
    # 축 레이블 제거
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    return fig

# ======================================================================
# 경로 / GitHub 설정
# ======================================================================
BASE_DIR   = "/tmp"
SESS_DIR   = os.path.join(BASE_DIR, "sessions")
os.makedirs(SESS_DIR, exist_ok=True)

GITHUB_TOKEN  = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO   = st.secrets.get("GITHUB_REPO", "")
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")

KST = timezone(timedelta(hours=9))
def now_kst() -> datetime: return datetime.now(tz=KST)
def to_iso_kst(dt: datetime) -> str:
    if dt.tzinfo is None: dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST).isoformat(timespec="seconds")
def kst_to_rfc3339_utc(dt_kst: datetime) -> str:
    if dt_kst.tzinfo is None: dt_kst = dt_kst.replace(tzinfo=KST)
    return dt_kst.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

# ======================================================================
# 키/상수
# ======================================================================
_YT_FALLBACK, _GEM_FALLBACK = [], []
YT_API_KEYS       = list(st.secrets.get("YT_API_KEYS", [])) or _YT_FALLBACK
GEMINI_API_KEYS   = list(st.secrets.get("GEMINI_API_KEYS", [])) or _GEM_FALLBACK
GEMINI_MODEL      = "gemini-2.5-flash-lite"
GEMINI_TIMEOUT    = 120
GEMINI_MAX_TOKENS = 2048
MAX_TOTAL_COMMENTS   = 120_000
MAX_COMMENTS_PER_VID = 4_000

# ======================================================================
# 세션 상태
# ======================================================================
def ensure_state():
    defaults = {
        "chat": [],
        "last_schema": None,
        "last_csv": "",
        "last_df": None,
        "sample_text": "",
        "loaded_session_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
ensure_state()

# ======================================================================
# GitHub API
# ======================================================================
def _gh_headers(token: str):
    h = {"Accept": "application/vnd.github+json"}
    if token: h["Authorization"] = f"token {token}"
    return h

def github_upload_file(repo, branch, path_in_repo, local_path, token):
    url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
    with open(local_path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")
    headers = _gh_headers(token)
    get_resp = requests.get(url + f"?ref={branch}", headers=headers)
    sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None
    data = {"message": f"archive {os.path.basename(path_in_repo)}", "content": content, "branch": branch}
    if sha: data["sha"] = sha
    resp = requests.put(url, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()

def github_list_dir(repo, branch, folder, token):
    url = f"https://api.github.com/repos/{repo}/contents/{folder}?ref={branch}"
    resp = requests.get(url, headers=_gh_headers(token))
    if resp.status_code != 200: return []
    return [x["name"] for x in resp.json() if isinstance(x, dict) and x.get("type") == "dir"]

def github_download_file(repo, branch, path_in_repo, token, local_path):
    url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}?ref={branch}"
    resp = requests.get(url, headers=_gh_headers(token))
    if resp.status_code == 200:
        data = resp.json()
        content = base64.b64decode(data["content"])
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(content)
        return True
    return False

def github_delete_folder(repo, branch, folder_path, token):
    contents_url = f"https://api.github.com/repos/{repo}/contents/{folder_path}?ref={branch}"
    headers = _gh_headers(token)
    resp = requests.get(contents_url, headers=headers)
    if resp.status_code != 200: return
    for item in resp.json():
        delete_url = f"https://api.github.com/repos/{repo}/contents/{item['path']}"
        data = {"message": f"delete: {item['name']}", "sha": item['sha'], "branch": branch}
        requests.delete(delete_url, headers=headers, json=data).raise_for_status()

def github_rename_session(old_name, new_name, token):
    contents_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/sessions/{old_name}?ref={GITHUB_BRANCH}"
    resp = requests.get(contents_url, headers=_gh_headers(token))
    resp.raise_for_status()
    files_to_move = resp.json()
    local_dir = os.path.join(SESS_DIR, "_tmp_move")
    os.makedirs(local_dir, exist_ok=True)
    for item in files_to_move:
        filename = item["name"]
        lp = os.path.join(local_dir, filename)
        github_download_file(GITHUB_REPO, GITHUB_BRANCH, item["path"], token, lp)
        github_upload_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{new_name}/{filename}", lp, token)
    github_delete_folder(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{old_name}", token)

# ======================================================================
# 세션 관리
# ======================================================================
def _build_session_name() -> str:
    if st.session_state.get("loaded_session_name"):
        return st.session_state.loaded_session_name
    schema = st.session_state.get("last_schema", {}) or {}
    kw = (schema.get("keywords", ["NoKeyword"]))[0]
    kw_slug = re.sub(r'[^\w-]', '', kw.replace(' ', '_'))[:20]
    return f"{kw_slug}_{now_kst().strftime('%Y-%m-%d_%H%M')}"

def save_current_session_to_github():
    if not all([GITHUB_REPO, GITHUB_TOKEN, st.session_state.chat, st.session_state.last_csv]):
        return False, "저장할 데이터가 없거나 GitHub 설정이 누락되었습니다."
    sess_name = _build_session_name()
    local_dir = os.path.join(SESS_DIR, sess_name)
    os.makedirs(local_dir, exist_ok=True)
    try:
        meta_path = os.path.join(local_dir, "qa.json")
        meta_data = {
            "chat": st.session_state.chat,
            "last_schema": st.session_state.last_schema,
            "sample_text": st.session_state.sample_text
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        comments_path = os.path.join(local_dir, "comments.csv")
        videos_path   = os.path.join(local_dir, "videos.csv")
        os.system(f'cp "{st.session_state.last_csv}" "{comments_path}"')
        if st.session_state.last_df is not None:
            st.session_state.last_df.to_csv(videos_path, index=False, encoding="utf-8-sig")
        github_upload_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/qa.json", meta_path, GITHUB_TOKEN)
        github_upload_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/comments.csv", comments_path, GITHUB_TOKEN)
        if os.path.exists(videos_path):
            github_upload_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/videos.csv", videos_path, GITHUB_TOKEN)
        st.session_state.loaded_session_name = sess_name
        return True, sess_name
    except Exception as e:
        return False, f"저장 실패: {e}"

def load_session_from_github(sess_name: str):
    with st.spinner(f"세션 '{sess_name}' 불러오는 중..."):
        try:
            local_dir = os.path.join(SESS_DIR, sess_name)
            qa_ok = github_download_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/qa.json", GITHUB_TOKEN, os.path.join(local_dir, "qa.json"))
            comments_ok = github_download_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/comments.csv", GITHUB_TOKEN, os.path.join(local_dir, "comments.csv"))
            videos_ok = github_download_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/videos.csv", GITHUB_TOKEN, os.path.join(local_dir, "videos.csv"))
            if not (qa_ok and comments_ok):
                st.error("세션 핵심 파일을 불러오는 데 실패했습니다."); return
            meta = json.load(open(os.path.join(local_dir, "qa.json"), encoding="utf-8"))
            st.session_state.clear(); ensure_state()
            st.session_state.update({
                "chat": meta.get("chat", []),
                "last_schema": meta.get("last_schema", None),
                "last_csv": os.path.join(local_dir, "comments.csv"),
                "last_df": pd.read_csv(os.path.join(local_dir, "videos.csv")) if videos_ok and os.path.exists(os.path.join(local_dir, "videos.csv")) else pd.DataFrame(),
                "loaded_session_name": sess_name,
                "sample_text": meta.get("sample_text", "")
            })
        except Exception as e:
            st.error(f"세션 로드 실패: {e}")

# 세션 로드/삭제/이름변경 트리거
if 'session_to_load' in st.session_state:
    load_session_from_github(st.session_state.pop('session_to_load')); st.rerun()
if 'session_to_delete' in st.session_state:
    sess_name = st.session_state.pop('session_to_delete')
    with st.spinner(f"세션 '{sess_name}' 삭제 중..."): github_delete_folder(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}", GITHUB_TOKEN)
    st.success("세션 삭제 완료."); time.sleep(1); st.rerun()
if 'session_to_rename' in st.session_state:
    old, new = st.session_state.pop('session_to_rename')
    if old and new and old != new:
        with st.spinner("이름 변경 중..."):
            try: github_rename_session(old, new, GITHUB_TOKEN); st.success("이름 변경 완료!")
            except Exception as e: st.error(f"변경 실패: {e}")
        time.sleep(1); st.rerun()

# ======================================================================
# 사이드바 UI
# ======================================================================
with st.sidebar:
    st.markdown(
        '<h2 style="font-weight:600; font-size:1.6rem; margin-bottom:1.5rem; '
        'background:-webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570, #F2A60C); '
        '-webkit-background-clip:text; -webkit-text-fill-color:transparent;">'
        '💬 유튜브 댓글분석: AI 챗봇</h2>',
        unsafe_allow_html=True
    )
    st.caption("문의: 미디어)디지털마케팅 데이터파트")

    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("✨ 새 채팅", use_container_width=True):
        st.session_state.clear(); ensure_state(); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.chat and st.session_state.last_csv:
        if st.button("💾 현재 대화 저장", use_container_width=True):
            with st.spinner("세션 저장 중..."):
                ok, msg = save_current_session_to_github()
            if ok: st.success(f"'{msg}' 저장 완료!"); time.sleep(2); st.rerun()
            else: st.error(msg)

    st.markdown("---")
    st.markdown("#### 대화 기록")

    if not all([GITHUB_TOKEN, GITHUB_REPO]):
        st.caption("GitHub 설정이 Secrets에 없습니다.")
    else:
        try:
            sessions = sorted(github_list_dir(GITHUB_REPO, GITHUB_BRANCH, "sessions", GITHUB_TOKEN), reverse=True)
            if not sessions: st.caption("저장된 기록이 없습니다.")
            else:
                editing_session = st.session_state.get("editing_session", None)
                st.markdown('<div class="session-list">', unsafe_allow_html=True)
                for sess in sessions:
                    if sess == editing_session:
                        new_name = st.text_input("새 이름:", value=sess, key=f"new_name_{sess}")
                        c1, c2 = st.columns(2)
                        if c1.button("✅", key=f"save_{sess}"):
                            st.session_state.session_to_rename = (sess, new_name); st.session_state.pop('editing_session', None); st.rerun()
                        if c2.button("❌", key=f"cancel_{sess}"):
                            st.session_state.pop('editing_session', None); st.rerun()
                    else:
                        c1, c2, c3 = st.columns([0.7, 0.15, 0.15])
                        if c1.button(sess, key=f"sess_{sess}", use_container_width=True):
                            st.session_state.session_to_load = sess; st.rerun()
                        if c2.button("✏️", key=f"edit_{sess}"):
                            st.session_state.editing_session = sess; st.rerun()
                        if c3.button("🗑️", key=f"del_{sess}"):
                            st.session_state.session_to_delete = sess; st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception:
            st.error("기록 로딩 실패")

# ======================================================================
# 공용 유틸
# ======================================================================
def scroll_to_bottom():
    st_html(
        "<script>let m=document.querySelectorAll('.stChatMessage'); if(m.length>0){m[m.length-1].scrollIntoView({behavior:'smooth'});} </script>",
        height=0
    )

TITLE_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s+.*$")
HEADER_DUP_RE = re.compile(r"유튜브\s*댓글\s*분석.*", re.IGNORECASE)
def tidy_answer(md: str) -> str:
    if not md: return md
    lines = [l for l in md.splitlines() if not (TITLE_LINE_RE.match(l) or HEADER_DUP_RE.search(l))]
    cleaned, prev_blank = [], False
    for l in lines:
        is_blank = not l.strip()
        if is_blank and prev_blank: continue
        cleaned.append(l); prev_blank = is_blank
    return "\n".join(cleaned).strip()

# ======================================================================
# URL/ID 추출 & 텍스트 정리
# ======================================================================
YTB_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")
def extract_video_ids_from_text(text: str) -> list:
    if not text: return []
    ids = set()
    for m in re.finditer(r"https?://youtu\.be/([A-Za-z0-9_-]{11})", text): ids.add(m.group(1))
    for m in re.finditer(r"https?://(?:www\.)?youtube\.com/shorts/([A-Za-z0-9_-]{11})", text): ids.add(m.group(1))
    for m in re.finditer(r"https?://(?:www\.)?youtube\.com/watch\?[^ \n]+", text):
        url = m.group(0)
        try:
            qs = dict((kv.split("=", 1) + [""])[:2] for kv in url.split("?", 1)[1].split("&"))
            v = qs.get("v", "")
            if YTB_ID_RE.fullmatch(v): ids.add(v)
        except Exception: pass
    return list(ids)

def strip_urls(s: str) -> str:
    if not s: return ""
    s = re.sub(r"https?://\S+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# ======================================================================
# YouTube / Gemini 래퍼
# ======================================================================
class RotatingKeys:
    def __init__(self, keys, state_key: str, on_rotate=None):
        self.keys = [k.strip() for k in (keys or []) if isinstance(k, str) and k.strip()][:10]
        self.state_key = state_key; self.on_rotate = on_rotate
        idx = st.session_state.get(state_key, 0)
        self.idx = 0 if not self.keys else (idx % len(self.keys)); st.session_state[state_key] = self.idx
    def current(self): return self.keys[self.idx % len(self.keys)] if self.keys else None
    def rotate(self):
        if not self.keys: return
        self.idx = (self.idx + 1) % len(self.keys); st.session_state[self.state_key] = self.idx
        if callable(self.on_rotate): self.on_rotate(self.idx, self.current())

class RotatingYouTube:
    def __init__(self, keys, state_key="yt_key_idx"):
        self.rot = RotatingKeys(keys, state_key); self.service = None; self._build()
    def _build(self):
        key = self.rot.current()
        if not key: raise RuntimeError("YouTube API Key가 비어 있습니다.")
        self.service = build("youtube", "v3", developerKey=key)
    def execute(self, factory):
        try: return factory(self.service).execute()
        except HttpError as e:
            status = getattr(getattr(e, 'resp', None), 'status', None)
            msg = (getattr(e, 'content', b'').decode('utf-8', 'ignore') or '').lower()
            if status in (403, 429) and any(t in msg for t in ["quota","rate","limit"]) and len(YT_API_KEYS) > 1:
                self.rot.rotate(); self._build(); return factory(self.service).execute()
            raise

LIGHT_PROMPT = (
    f"역할: 유튜브 댓글 반응 분석기의 자연어 해석가.\n"
    f"목표: 한국어 입력에서 [기간(KST)]과 [키워드/엔티티/옵션]을 해석.\n"
    f"규칙:\n"
    f"- 기간은 Asia/Seoul 기준, 상대기간의 종료는 지금.\n"
    f"- '키워드'는 검색에 사용할 가장 핵심적인 주제 1개로 한정.\n"
    f"- 숫자+단위(24시간/48시간/7일/3주/12개월)는 '최근 N'으로 해석.\n"
    f"- 옵션: include_replies, channel_filter(any|official|unofficial), lang(ko|en|auto).\n\n"
    f"출력(6줄):\n"
    f"- 한 줄 요약: <문장>\n"
    f"- 기간(KST): <YYYY-MM-DDTHH:MM:SS+09:00> ~ <YYYY-MM-DDTHH:MM:SS+09:00>\n"
    f"- 키워드: [<핵심 키워드 1개>]\n"
    f"- 엔티티/보조: [<콤마로 구분>]\n"
    f"- 옵션: {{ include_replies: true|false, channel_filter: \"any|official|unofficial\", lang: \"ko|en|auto\" }}\n"
    f"- 원문: {{USER_QUERY}}\n\n"
    f"현재 KST: {to_iso_kst(now_kst())}\n"
    f"입력:\n{{USER_QUERY}}"
)

def call_gemini_rotating(model_name, keys, system_instruction, user_payload,
                         timeout_s=GEMINI_TIMEOUT, max_tokens=GEMINI_MAX_TOKENS) -> str:
    rk = RotatingKeys(keys, "gem_key_idx")
    if not rk.current(): raise RuntimeError("Gemini API Key가 비어 있습니다.")
    for _ in range(len(rk.keys) or 1):
        try:
            genai.configure(api_key=rk.current())
            model = genai.GenerativeModel(model_name, generation_config={"temperature":0.2,"max_output_tokens":max_tokens})
            resp = model.generate_content([system_instruction, user_payload], request_options={"timeout": timeout_s})
            out = getattr(resp, "text", None)
            if out: return out
            c0 = (getattr(resp, "candidates", None) or [None])[0]
            if c0 and getattr(c0, "content", None) and getattr(c0.content, "parts", None):
                p0 = c0.content.parts[0]
                if hasattr(p0, "text"): return p0.text
            return ""
        except Exception as e:
            if "429" in str(e).lower() and len(rk.keys) > 1:
                rk.rotate(); continue
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
        if ir := re.search(r"include_replies\s*:\s*(true|false)", blob, re.I): options["include_replies"] = (ir.group(1).lower()=="true")
        if cf := re.search(r"channel_filter\s*:\s*\"(any|official|unofficial)\"", blob, re.I): options["channel_filter"]=cf.group(1)
        if lg := re.search(r"lang\s*:\s*\"(ko|en|auto)\"", blob, re.I): options["lang"]=lg.group(1)
    # 기간 파싱 실패 시 최근 1주 자동 적용
    if not (start_iso and end_iso):
        end_dt = now_kst()
        start_dt = end_dt - timedelta(days=7)
        start_iso, end_iso = to_iso_kst(start_dt), to_iso_kst(end_dt)
    if not keywords:
        keywords = [m[0]] if (m := re.findall(r"[가-힣A-Za-z0-9]{2,}", raw)) else ["유튜브"]
    # 엔티티는 검색에 사용하지 않도록 유지(요청사항)
    return {"start_iso":start_iso, "end_iso":end_iso, "keywords":keywords[:1], "entities":entities, "options":options, "raw":raw}

def yt_search_videos(rt, keyword, max_results, order="relevance", published_after=None, published_before=None):
    video_ids, token = [], None
    while len(video_ids) < max_results:
        params = {"q": keyword, "part":"id", "type":"video", "order":order, "maxResults":min(50, max_results-len(video_ids))}
        if published_after: params["publishedAfter"] = published_after
        if published_before: params["publishedBefore"] = published_before
        if token: params["pageToken"] = token
        resp = rt.execute(lambda s: s.search().list(**params))
        for it in resp.get("items", []):
            vid = it["id"]["videoId"]
            if vid not in video_ids: video_ids.append(vid)
        token = resp.get("nextPageToken")
        if not token: break
        time.sleep(0.25)
    return video_ids

def yt_video_statistics(rt, video_ids):
    rows = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        if not batch: continue
        resp = rt.execute(lambda s: s.videos().list(part="statistics,snippet,contentDetails", id=",".join(batch)))
        for item in resp.get("items", []):
            stats = item.get("statistics", {}); snip = item.get("snippet", {}); cont = item.get("contentDetails", {})
            dur = cont.get("duration", "")
            h = re.search(r"(\d+)H", dur); m = re.search(r"(\d+)M", dur); s = re.search(r"(\d+)S", dur)
            dur_sec = (int(h.group(1)) * 3600 if h else 0) + (int(m.group(1)) * 60 if m else 0) + (int(s.group(1)) if s else 0)
            vid_id = item.get("id")
            rows.append({
                "video_id": vid_id, "video_url": f"https://www.youtube.com/watch?v={vid_id}",
                "title": snip.get("title", ""), "channelTitle": snip.get("channelTitle", ""),
                "publishedAt": snip.get("publishedAt", ""), "duration": dur,
                "shortType": "Shorts" if dur_sec <= 60 else "Clip",
                "viewCount": int(stats.get("viewCount", 0) or 0),
                "likeCount": int(stats.get("likeCount", 0) or 0),
                "commentCount": int(stats.get("commentCount", 0) or 0),
            })
        time.sleep(0.25)
    return rows

def yt_all_replies(rt, parent_id, video_id, title="", short_type="Clip", cap=None):
    replies, token = [], None
    while not (cap is not None and len(replies) >= cap):
        try:
            resp = rt.execute(lambda s: s.comments().list(part="snippet", parentId=parent_id, maxResults=100, pageToken=token, textFormat="plainText"))
        except HttpError: break
        for c in resp.get("items", []):
            sn = c["snippet"]
            replies.append({
                "video_id": video_id, "video_title": title, "shortType": short_type,
                "comment_id": c.get("id",""), "parent_id": parent_id, "isReply": 1,
                "author": sn.get("authorDisplayName",""), "text": sn.get("textDisplay","") or "",
                "publishedAt": sn.get("publishedAt",""), "likeCount": int(sn.get("likeCount",0) or 0),
            })
        token = resp.get("nextPageToken")
        if not token: break
        time.sleep(0.2)
    return replies[:cap] if cap is not None else replies

def yt_all_comments_sync(rt, video_id, title="", short_type="Clip", include_replies=True, max_per_video=None):
    rows, token = [], None
    while not (max_per_video is not None and len(rows) >= max_per_video):
        try:
            resp = rt.execute(lambda s: s.commentThreads().list(part="snippet,replies", videoId=video_id, maxResults=100, pageToken=token, textFormat="plainText"))
        except HttpError: break
        for it in resp.get("items", []):
            top = it["snippet"]["topLevelComment"]["snippet"]
            thread_id = it["snippet"]["topLevelComment"]["id"]
            rows.append({
                "video_id": video_id, "video_title": title, "shortType": short_type,
                "comment_id": thread_id, "parent_id": "", "isReply": 0,
                "author": top.get("authorDisplayName",""), "text": top.get("textDisplay","") or "",
                "publishedAt": top.get("publishedAt",""), "likeCount": int(top.get("likeCount",0) or 0),
            })
            if include_replies and int(it["snippet"].get("totalReplyCount", 0) or 0) > 0:
                cap = None if max_per_video is None else max(0, max_per_video - len(rows))
                if cap == 0: break
                rows.extend(yt_all_replies(rt, thread_id, video_id, title, short_type, cap=cap))
        token = resp.get("nextPageToken")
        if not token: break
        time.sleep(0.2)
    return rows[:max_per_video] if max_per_video is not None else rows

def parallel_collect_comments_streaming(video_list, rt_keys, include_replies, max_total_comments, max_per_video, prog_bar):
    out_csv = os.path.join(BASE_DIR, f"collect_{uuid4().hex}.csv")
    wrote_header, total_written, done, total_videos = False, 0, 0, len(video_list)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            ex.submit(yt_all_comments_sync, RotatingYouTube(rt_keys), v["video_id"], v.get("title",""), v.get("shortType","Clip"), include_replies, max_per_video): v
            for v in video_list
        }
        for f in as_completed(futures):
            try:
                comm = f.result()
                if comm:
                    dfc = pd.DataFrame(comm)
                    dfc.to_csv(out_csv, index=False, mode=("a" if wrote_header else "w"), header=(not wrote_header), encoding="utf-8-sig")
                    wrote_header = True; total_written += len(dfc)
            except Exception:
                pass
            done += 1
            prog_bar.progress(min(0.90, 0.50 + (done/total_videos)*0.40 if total_videos>0 else 0.50), text="댓글 수집중…")
            if total_written >= max_total_comments: break
    return out_csv, total_written

def serialize_comments_for_llm_from_file(csv_path: str, max_chars_per_comment=280, max_total_chars=420_000):
    if not os.path.exists(csv_path): return "", 0, 0
    try: df_all = pd.read_csv(csv_path)
    except Exception: return "", 0, 0
    if df_all.empty: return "", 0, 0
    df_top = df_all.sort_values("likeCount", ascending=False).head(1000)
    remain = df_all.drop(df_top.index)
    df_rand = remain.sample(n=min(1000, len(remain))) if not remain.empty else pd.DataFrame()
    df_sample = pd.concat([df_top, df_rand])
    lines, total = [], 0
    for _, r in df_sample.iterrows():
        if total >= max_total_chars: break
        text = str(r.get("text","") or "").replace("\n"," ")
        prefix = f"[{'R' if int(r.get('isReply',0))==1 else 'T'}|♥{int(r.get('likeCount',0))}] {str(r.get('author','')).replace('\n',' ')}: "
        body = text[:max_chars_per_comment] + '…' if len(text) > max_chars_per_comment else text
        line = prefix + body
        if total + len(line) + 1 > max_total_chars: break
        lines.append(line); total += len(line) + 1
    return "\n".join(lines), len(lines), total

# ======================================================================
# 메타/다운로드 + 채팅 렌더
# ======================================================================
def render_metadata_and_downloads():
    schema = st.session_state.get("last_schema")
    if not schema: return
    kw_main = schema.get("keywords", [])
    start_iso = schema.get("start_iso",""); end_iso = schema.get("end_iso","")
    try:
        start_dt_str = datetime.fromisoformat(start_iso).astimezone(KST).strftime('%Y-%m-%d %H:%M')
        end_dt_str   = datetime.fromisoformat(end_iso).astimezone(KST).strftime('%Y-%m-%d %H:%M')
    except (ValueError, TypeError):
        start_dt_str = start_iso.split('T')[0] if start_iso else ""
        end_dt_str   = end_iso.split('T')[0] if end_iso else ""
    with st.container(border=True):
        st.markdown(
            f"<div style='font-size:14px; color:#4b5563; line-height:1.8;'>"
            f"<span style='font-weight:600;'>키워드:</span> {', '.join(kw_main) if kw_main else '(없음)'}<br>"
            f"<span style='font-weight:600;'>기간:</span> {start_dt_str} ~ {end_dt_str} (KST)"
            f"</div>", unsafe_allow_html=True
        )
        csv_path  = st.session_state.get("last_csv"); df_videos = st.session_state.get("last_df")
        if csv_path and os.path.exists(csv_path) and df_videos is not None and not df_videos.empty:
            with open(csv_path, "rb") as f: comment_csv_data = f.read()
            buf = io.BytesIO(); df_videos.to_csv(buf, index=False, encoding="utf-8-sig"); video_csv_data = buf.getvalue()
            keywords_str = "_".join(kw_main).replace(" ","_") if kw_main else "data"
            now_str = now_kst().strftime('%Y%m%d')
            col1, col2, col3, _ = st.columns([1.1, 1.2, 1.2, 6.5])
            col1.markdown("<div style='font-size:14px; color:#4b5563; font-weight:600; padding-top:5px;'>다운로드:</div>", unsafe_allow_html=True)
            with col2: st.download_button("전체댓글", comment_csv_data, f"comments_{keywords_str}_{now_str}.csv", "text/csv")
            with col3: st.download_button("영상목록",  video_csv_data, f"videos_{keywords_str}_{now_str}.csv",   "text/csv")

def render_chat():
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ======================================================================
# 정량 시각화: 1 2 / 3 4 레이아웃 (Treemap / Timeseries / Top Videos / Top Authors)
# ======================================================================
def _read_comments_lazy(csv_path: str, usecols=None, chunksize=200_000):
    if not csv_path or not os.path.exists(csv_path): return []
    try:
        return pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize)
    except Exception:
        return []

def compute_keyword_counts(csv_path: str, per_comment_cap: int = 200, topn: int = 50):
    # 간단 토큰: 공백 기준 split, 한글/영문/숫자 2자 이상만
    counts = {}
    for chunk in _read_comments_lazy(csv_path, usecols=["text"]):
        texts = chunk["text"].astype(str).str.slice(0, per_comment_cap).tolist()
        for t in texts:
            # 간이 토큰 (형태소기반은 무거우니 경량)
            tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", t)
            for w in tokens:
                counts[w] = counts.get(w, 0) + 1
    if not counts: return pd.DataFrame(columns=["word","count"])
    s = pd.Series(counts).sort_values(ascending=False).head(topn)
    return s.rename_axis("word").reset_index(name="count")

def timeseries_from_file(csv_path: str):
    # 시간/일자별 댓글량
    tmin = None; tmax = None
    for chunk in _read_comments_lazy(csv_path, usecols=["publishedAt"]):
        dt = pd.to_datetime(chunk["publishedAt"], errors="coerce", utc=True)
        if dt.notna().any():
            lo, hi = dt.min(), dt.max()
            tmin = lo if (tmin is None or (lo < tmin)) else tmin
            tmax = hi if (tmax is None or (hi > tmax)) else tmax
    if tmin is None or tmax is None: return None, None
    span_hours = (tmax - tmin).total_seconds()/3600.0
    use_hour = (span_hours <= 48)
    agg = {}
    for chunk in _read_comments_lazy(csv_path, usecols=["publishedAt"]):
        dt = pd.to_datetime(chunk["publishedAt"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")
        dt = dt.dropna()
        if dt.empty: continue
        bucket = (dt.dt.floor("H") if use_hour else dt.dt.floor("D"))
        vc = bucket.value_counts()
        for t, c in vc.items(): agg[t] = agg.get(t, 0) + int(c)
    ts = pd.Series(agg).sort_index().rename("count").reset_index().rename(columns={"index":"bucket"})
    return ts, ("시간별" if use_hour else "일자별")

def top_authors_from_file(csv_path: str, topn=10):
    counts = {}
    for chunk in _read_comments_lazy(csv_path, usecols=["author"]):
        vc = chunk["author"].astype(str).value_counts()
        for k, v in vc.items():
            counts[k] = counts.get(k, 0) + int(v)
    if not counts: return pd.DataFrame(columns=["author","count"])
    s = pd.Series(counts).sort_values(ascending=False).head(topn)
    return s.rename_axis("author").reset_index(name="count")

def render_quant_viz_from_paths(comments_csv_path: str, df_stats: pd.DataFrame, scope_label="(KST)"):
    if not comments_csv_path or not os.path.exists(comments_csv_path): return

    # 1 2
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("① 키워드 트리맵")
            # 히트맵 스타일(단일 팔레트 진하기): Blues
            kw_df = compute_keyword_counts(comments_csv_path, per_comment_cap=200, topn=40)
            if kw_df.empty:
                st.info("키워드가 충분치 않습니다.")
            else:
                # normalize for color intensity only
                maxc = kw_df["count"].max() if not kw_df.empty else 1
                kw_df["color_val"] = kw_df["count"] / maxc
                # 단일 팔레트: px.colors.sequential.Blues를 사용하되, 하나의 컬러 스케일에 매핑
                fig_tm = px.treemap(
                    kw_df,
                    path=["word"],
                    values="count",
                    color="color_val",
                    color_continuous_scale=px.colors.sequential.Blues,
                )
                fig_tm.update_traces(root_color="lightgray")
                fig_tm.update_layout(coloraxis_showscale=False)
                fig_tm = _small_fig(fig_tm, height=320, title_size=14, font_size=12, legend_size=11, margin=(0,0,24,0))
                st.plotly_chart(fig_tm, use_container_width=True)

    with c2:
        with st.container(border=True):
            st.subheader(f"② 시점별 댓글량 추이 {scope_label}")
            ts, label = timeseries_from_file(comments_csv_path)
            if ts is not None and not ts.empty:
                fig_ts = px.line(ts, x="bucket", y="count", markers=True, title=f"{label} 댓글량")
                fig_ts.update_layout(showlegend=False)
                fig_ts = _small_fig(fig_ts, height=320)
                st.plotly_chart(fig_ts, use_container_width=True)
            else:
                st.info("댓글 타임스탬프가 비어 있습니다.")

    # 3 4
    c3, c4 = st.columns(2)
    with c3:
        with st.container(border=True):
            st.subheader("③ Top10 영상 (댓글수)")
            if df_stats is not None and not df_stats.empty and "commentCount" in df_stats.columns:
                top_vids = df_stats.sort_values(by="commentCount", ascending=False).head(10).copy()
                top_vids["title_short"] = top_vids["title"].apply(lambda t: t[:20] + "…" if isinstance(t,str) and len(t)>20 else t)
                fig_vids = px.bar(top_vids, x="commentCount", y="title_short", orientation="h", text="commentCount", title="영상별 댓글수")
                # hover에 링크 노출(클릭은 브라우저 제한으로 불가)
                fig_vids.update_traces(customdata=np.stack([top_vids["video_url"]], axis=-1))
                fig_vids.update_traces(hovertemplate="댓글수: %{x}<br>제목: %{y}<br>URL: %{customdata[0]}<extra></extra>")
                fig_vids.update_layout(showlegend=False)
                fig_vids = _small_fig(fig_vids, height=320)
                st.plotly_chart(fig_vids, use_container_width=True)
            else:
                st.info("영상 메타가 없습니다.")

    with c4:
        with st.container(border=True):
            st.subheader("④ 댓글 작성자 Top10")
            ta = top_authors_from_file(comments_csv_path, topn=10)
            if ta is not None and not ta.empty:
                fig_auth = px.bar(ta, x="count", y="author", orientation="h", text="count", title="작성자 활동량")
                fig_auth.update_layout(showlegend=False)
                fig_auth = _small_fig(fig_auth, height=320)
                st.plotly_chart(fig_auth, use_container_width=True)
            else:
                st.info("작성자 데이터 없음")

# ======================================================================
# 파이프라인 (첫 질문 / 후속 질문)
# ======================================================================
def run_pipeline_first_turn(user_query: str, extra_video_ids=None, only_these_videos: bool=False):
    extra_video_ids = list(dict.fromkeys(extra_video_ids or []))
    prog_bar = st.progress(0, text="준비 중…")

    if not GEMINI_API_KEYS: return "오류: Gemini API Key가 설정되지 않았습니다."
    prog_bar.progress(0.05, text="해석중…")
    light = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, "", LIGHT_PROMPT.replace("{USER_QUERY}", user_query))
    schema = parse_light_block_to_schema(light)
    st.session_state["last_schema"] = schema

    if not YT_API_KEYS: return "오류: YouTube API Key가 설정되지 않았습니다."
    prog_bar.progress(0.10, text="영상 수집중…")

    rt = RotatingYouTube(YT_API_KEYS)
    start_dt = datetime.fromisoformat(schema["start_iso"]); end_dt = datetime.fromisoformat(schema["end_iso"])
    kw_main  = schema.get("keywords", [])[:1]  # 메인 키워드 1개만
    # 엔티티는 검색에 사용하지 않음 (요청사항 반영)

    if only_these_videos and extra_video_ids:
        all_ids = extra_video_ids
    else:
        all_ids = []
        for base_kw in (kw_main or ["유튜브"]):
            all_ids.extend(yt_search_videos(rt, base_kw, 60, "relevance", kst_to_rfc3339_utc(start_dt), kst_to_rfc3339_utc(end_dt)))
        if extra_video_ids:
            all_ids.extend(extra_video_ids)
    all_ids = list(dict.fromkeys(all_ids))

    prog_bar.progress(0.40, text="댓글 수집 준비중…")
    df_stats = pd.DataFrame(yt_video_statistics(rt, all_ids))
    st.session_state["last_df"] = df_stats

    csv_path, total_cnt = parallel_collect_comments_streaming(
        df_stats.to_dict('records'), YT_API_KEYS,
        bool(schema.get("options", {}).get("include_replies")),
        MAX_TOTAL_COMMENTS, MAX_COMMENTS_PER_VID, prog_bar
    )
    st.session_state["last_csv"] = csv_path

    if total_cnt == 0:
        prog_bar.empty(); return "지정 조건에서 댓글을 찾을 수 없습니다. 다른 조건으로 시도해 보세요."

    prog_bar.progress(0.90, text="AI 분석중…")
    sample_text, _, _ = serialize_comments_for_llm_from_file(csv_path)
    st.session_state["sample_text"] = sample_text

    sys = (
        "너는 유튜브 댓글을 분석하는 어시스턴트다. 먼저 [사용자 원본 질문]을 확인하여 "
        "핵심 관점을 파악한 뒤, 댓글 샘플을 바탕으로 핵심 포인트를 항목화하고 "
        "긍/부/중 비율과 대표 코멘트(10개 미만)를 제시하라. 반복 금지."
    )
    payload = (
        f"[사용자 원본 질문]: {user_query}\n\n"
        f"[키워드]: {', '.join(kw_main)}\n"
        f"[기간(KST)]: {schema['start_iso']} ~ {schema['end_iso']}\n\n"
        f"[댓글 샘플]:\n{sample_text}\n"
    )
    answer_md_raw = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, sys, payload)

    prog_bar.progress(1.0, text="완료"); time.sleep(0.3); prog_bar.empty(); gc.collect()
    return tidy_answer(answer_md_raw)

def run_followup_turn(user_query: str):
    schema = st.session_state.get("last_schema")
    if not schema: return "오류: 이전 분석 기록이 없습니다. 새 채팅을 시작해주세요."
    sample_text = st.session_state.get("sample_text", "")
    context = "\n".join(f"[이전 {'Q' if m['role']=='user' else 'A'}]: {m['content']}" for m in st.session_state["chat"][-10:])
    sys = (
        "너는 사용자의 질문 의도를 정확히 파악하여 핵심만 답하는 유튜브 댓글 분석 챗봇이다.\n"
        "정성 질문엔 요약+코멘트 예시, 정량 질문엔 숫자 중심으로 간결히."
    )
    payload = f"{context}\n\n[현재 질문]: {user_query}\n[기간(KST)]: {schema.get('start_iso','?')} ~ {schema.get('end_iso','?')}\n\n[댓글 샘플]:\n{sample_text}\n"
    with st.spinner("💬 AI가 답변을 구성 중입니다..."):
        response_raw = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, sys, payload)
    return tidy_answer(response_raw)

# ======================================================================
# 메인: 인트로 / 메타 / 채팅 / (차트 슬롯)
# ======================================================================
if not st.session_state.chat:
    st.markdown(
        """
<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; height:70vh;">
  <h1 style="font-size:3.2rem; font-weight:600;
             background:-webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570, #F2A60C);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    유튜브 댓글분석: AI 챗봇
  </h1>
  <p style="font-size:1.1rem; color:#4b5563;">관련영상 유튜브 댓글반응을 AI가 요약해줍니다</p>
  <div style="margin-top:2rem; padding:1rem 1.5rem; border:1px solid #e5e7eb; border-radius:12px;
              background-color:#fafafa; max-width:600px; text-align:left;">
    <h4 style="margin-bottom:0.6rem; font-weight:600;">⚠️ 사용 주의사항</h4>
    <ol style="padding-left:20px; margin:0;">
      <li><strong>첫 질문 시</strong> 댓글 수집 및 AI 분석에 시간이 걸릴 수 있어요.</li>
      <li>한 세션에서는 <strong>하나의 주제</strong>만 다루는 것을 권장해요.</li>
      <li>첫 질문에는 <strong>기간</strong>을 명시해 주세요. (예: 최근 48시간 / 5월 1일부터)</li>
    </ol>
  </div>
</div>
""",
        unsafe_allow_html=True
    )
else:
    # ① 메타/다운로드
    render_metadata_and_downloads()

    # ② 시각화(항상 메타/다운로드 바로 아래, 채팅 위에 고정)
    if st.session_state.get("last_csv"):
        with st.container(border=True):
            render_quant_viz_from_paths(
                st.session_state["last_csv"],
                st.session_state.get("last_df"),
                scope_label="(KST)"
            )

    # ③ 채팅
    render_chat()
    scroll_to_bottom()


# ======================================================================
# 입력창 + 처리
# ======================================================================
prompt = st.chat_input("예) 최근 7일 태풍상사 반응 요약해줘 / 또는 유튜브 URL 붙여도 OK")
if prompt:
    st.session_state.chat.append({"role":"user","content":prompt})
    st.rerun()

if st.session_state.chat and st.session_state.chat[-1]["role"] == "user":
    user_query = st.session_state.chat[-1]["content"]
    url_ids    = extract_video_ids_from_text(user_query)
    natural    = strip_urls(user_query)
    has_urls   = len(url_ids) > 0
    has_nat    = len(natural) > 0

    if not st.session_state.get("last_csv"):
        if has_urls and not has_nat:
            response = run_pipeline_first_turn(user_query, extra_video_ids=url_ids, only_these_videos=True)
        elif has_urls and has_nat:
            response = run_pipeline_first_turn(user_query, extra_video_ids=url_ids, only_these_videos=False)
        else:
            response = run_pipeline_first_turn(user_query)
    else:
        response = run_followup_turn(user_query)

    st.session_state.chat.append({"role":"assistant","content":response})


    st.rerun()
