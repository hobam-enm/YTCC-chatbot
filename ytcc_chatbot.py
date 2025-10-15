# -*- coding: utf-8 -*-
# 💬 유튜브 댓글분석기 — 순수 챗봇 모드 (메타 1회 표시 / 단일 로딩바 / 자동 스크롤 / 핵심만 응답)
# - 첫 질문: 자연어 해석 → 영상 수집 → 댓글 수집(스트리밍 CSV) → AI요약 (단일 진행바)
# - 후속 질문: 재수집 없음(기존 샘플+대화 맥락만으로 답변)
# - 정량/다운로드/중간 로그 전부 제거. 채팅만.

import streamlit as st
import pandas as pd
import os, re, gc, time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4
import io # CSV 다운로드 인코딩을 위한 모듈 추가

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai
from streamlit.components.v1 import html as st_html

# -------------------- 페이지/전역 --------------------
st.set_page_config(page_title="유튜브 댓글분석: 챗봇", layout="wide", initial_sidebar_state="expanded")

# [수정] 챗봇 UI 스타일
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

/* 채팅 메시지 기본 스타일 */
[data-testid="stChatMessage"] {
    width: fit-content;
    margin-bottom: 1rem;
    padding: 0.8rem 1rem;
    border-radius: 18px;
    line-height: 1.5;
}

/* AI 답변 (assistant) 스타일 - 너비 제한 없음 */
[data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) {
    max-width: none; /* [수정] 너비 제한 제거 */
    background-color: #f0f2f6;
    border: 1px solid #d1d5db;
}

/* AI 답변 내부 텍스트 스타일 */
[data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) p,
[data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) li,
[data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) ol,
[data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) ul,
[data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) code {
    font-size: 0.9rem;
    color: #202123;
}

/* 사용자 질문 (user) 스타일 */
[data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-user"]) {
    max-width: 90%; /* 사용자 질문은 너비 제한 유지 */
    background-color: #0084ff;
    color: white;
    margin-left: auto;
}
[data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-user"]) p,
[data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-user"]) li {
    color: white;
}
</style>
""", unsafe_allow_html=True)


BASE_DIR = "/tmp"; os.makedirs(BASE_DIR, exist_ok=True)
KST = timezone(timedelta(hours=9))
def now_kst(): return datetime.now(tz=KST)
def to_iso_kst(dt: datetime) -> str:
    if dt.tzinfo is None: dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST).isoformat(timespec="seconds")
def kst_to_rfc3339_utc(dt_kst: datetime) -> str:
    if dt_kst.tzinfo is None: dt_kst = dt_kst.replace(tzinfo=KST)
    return dt_kst.astimezone(timezone.utc).isoformat().replace("+00:00","Z")

# -------------------- 키/상수 --------------------
_YT_FALLBACK = []
_GEM_FALLBACK = []
YT_API_KEYS       = list(st.secrets.get("YT_API_KEYS", [])) or _YT_FALLBACK
GEMINI_API_KEYS   = list(st.secrets.get("GEMINI_API_KEYS", [])) or _GEM_FALLBACK
GEMINI_MODEL      = st.secrets.get("GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_TIMEOUT    = int(st.secrets.get("GEMINI_TIMEOUT", 120))
GEMINI_MAX_TOKENS = int(st.secrets.get("GEMINI_MAX_TOKENS", 2048))

MAX_TOTAL_COMMENTS   = 120_000
MAX_COMMENTS_PER_VID = 4_000

# -------------------- 상태 --------------------
def ensure_state():
    defaults = dict(
        chat=[],
        last_schema=None,
        last_csv="",
        last_df=None,
        sample_text="",
    )
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
ensure_state()

# -------------------- 사이드바 (수정 없음) --------------------
with st.sidebar:
    st.markdown("""
    <style>
        [data-testid="stSidebarUserContent"] {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 4rem);
        }
        .contact-info {
            margin-top: auto;
        }
    </style>
    """, unsafe_allow_html=True)

    if st.button("✨ 새 채팅", use_container_width=True, type="secondary"):
        st.session_state.clear()
        st.rerun()

    st.markdown("""
    <div class="contact-info">
        <hr>
        <h3>📞 문의</h3>
        <p>미디어)디지털마케팅 데이터파트 김호범</p>
    </div>
    """, unsafe_allow_html=True)


# -------------------- 로직 (수정 없음) --------------------
def scroll_to_bottom():
    st_html("<script> let last_message = document.querySelectorAll('.stChatMessage'); if (last_message.length > 0) { last_message[last_message.length - 1].scrollIntoView(); } </script>", height=0)

def render_metadata_outside_chat():
    if not st.session_state.get("last_schema"): return
    schema = st.session_state["last_schema"]
    kw_main  = schema.get("keywords", [])
    start_iso, end_iso = schema.get('start_iso', ''), schema.get('end_iso', '')
    try:
        start_dt_str = datetime.fromisoformat(start_iso).astimezone(KST).strftime('%Y-%m-%d %H:%M')
        end_dt_str = datetime.fromisoformat(end_iso).astimezone(KST).strftime('%Y-%m-%d %H:%M')
    except (ValueError, TypeError):
        start_dt_str = start_iso.split('T')[0] if start_iso else ""
        end_dt_str = end_iso.split('T')[0] if end_iso else ""
    metadata_html = (
        f"<div style='font-size:14px; color:#4b5563; padding:8px 12px; border-radius:8px; border:1px solid #e5e7eb; margin-bottom:1rem; background-color: #f9fafb;'>"
        f"**📊 현재 분석 컨텍스트:**<br>"
        f"<span style='font-weight:600;'>키워드:</span> {', '.join(kw_main) if kw_main else '(없음)'}<br>"
        f"<span style='font-weight:600;'>기간:</span> {start_dt_str} ~ {end_dt_str} (KST)"
        f"</div>"
    )
    st.markdown(metadata_html, unsafe_allow_html=True)

class RotatingKeys:
    def __init__(self, keys, state_key: str, on_rotate=None):
        self.keys = [k.strip() for k in (keys or []) if isinstance(k, str) and k.strip()][:10]
        self.state_key = state_key
        self.on_rotate  = on_rotate
        idx = st.session_state.get(state_key, 0)
        self.idx = 0 if not self.keys else (idx % len(self.keys))
        st.session_state[state_key] = self.idx
    def current(self):
        return (self.keys[self.idx % len(self.keys)]) if self.keys else None
    def rotate(self):
        if not self.keys: return
        self.idx = (self.idx + 1) % len(self.keys)
        st.session_state[self.state_key] = self.idx
        if callable(self.on_rotate): self.on_rotate(self.idx, self.current())

class RotatingYouTube:
    def __init__(self, keys, state_key="yt_key_idx"):
        self.rot = RotatingKeys(keys, state_key)
        self.service = None
        self._build()
    def _build(self):
        key = self.rot.current()
        if not key: raise RuntimeError("YouTube API Key가 비어 있습니다.")
        self.service = build("youtube", "v3", developerKey=key)
    def execute(self, factory):
        try:
            return factory(self.service).execute()
        except HttpError as e:
            status = getattr(getattr(e,'resp',None),'status',None)
            msg = (getattr(e,'content',b'').decode('utf-8','ignore') or '').lower()
            quotaish = status in (403,429) and any(t in msg for t in ["quota","rate","limit"])
            if quotaish and len(YT_API_KEYS) > 1:
                self.rot.rotate(); self._build()
                return factory(self.service).execute()
            raise

LIGHT_PROMPT = (
    "역할: 유튜브 댓글 반응 분석기의 자연어 해석가.\n"
    "목표: 한국어 입력에서 [기간(KST)]과 [키워드/엔티티/옵션]을 해석.\n"
    "규칙:\n"
    "- 기간은 Asia/Seoul 기준, 상대기간의 종료는 지금.\n"
    "- 옵션 탐지: include_replies, channel_filter(any|official|unofficial), lang(ko|en|auto).\n\n"
    "출력(6줄 고정):\n"
    "- 한 줄 요약: <문장>\n"
    "- 기간(KST): <YYYY-MM-DDTHH:MM:SS+09:00> ~ <YYYY-MM-DDTHH:MM:SS+09:00>\n"
    "- 키워드: [<메인1>, <메인2>…]\n"
    "- 엔티티/보조: [<보조들>]\n"
    "- 옵션: { include_replies: true|false, channel_filter: \"any|official|unofficial\", lang: \"ko|en|auto\" }\n"
    "- 원문: {USER_QUERY}\n\n"
    f"현재 KST: {to_iso_kst(now_kst())}\n입력:\n{{USER_QUERY}}"
)

def is_gemini_quota_error(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return ("429" in msg) or ("too many requests" in msg) or ("rate limit" in msg) or ("resource exhausted" in msg) or ("quota" in msg)

def call_gemini_rotating(model_name, keys, system_instruction, user_payload, timeout_s=120, max_tokens=2048) -> str:
    rk = RotatingKeys(keys, "gem_key_idx")
    if not rk.current(): raise RuntimeError("Gemini API Key가 비어 있습니다.")
    attempts = 0
    while attempts < (len(rk.keys) if rk.keys else 1):
        try:
            genai.configure(api_key=rk.current())
            model = genai.GenerativeModel(model_name, generation_config={"temperature":0.2,"max_output_tokens":max_tokens})
            resp = model.generate_content([system_instruction, user_payload], request_options={"timeout": timeout_s})
            out = getattr(resp,"text",None)
            if not out and getattr(resp, "candidates", None):
                c0 = resp.candidates[0]
                if getattr(c0, "content", None) and getattr(c0.content, "parts", None):
                    p0 = c0.content.parts[0]
                    if hasattr(p0, "text"): out = p0.text
            return out or ""
        except Exception as e:
            if is_gemini_quota_error(e) and len(rk.keys) > 1:
                rk.rotate(); attempts += 1; continue
            raise

def parse_light_block_to_schema(light_text: str) -> dict:
    raw = (light_text or "").strip()
    m_time = re.search(r"기간\(KST\)\s*:\s*([^~]+)~\s*([^\n]+)", raw)
    start_iso = m_time.group(1).strip() if m_time else None
    end_iso   = m_time.group(2).strip() if m_time else None
    m_kw = re.search(r"키워드\s*:\s*\[(.*?)\]", raw, flags=re.DOTALL)
    keywords = [p.strip() for p in re.split(r"\s*,\s*", m_kw.group(1)) if p.strip()] if m_kw else []
    m_ent = re.search(r"엔티티/보조\s*:\s*\[(.*?)\]", raw, flags=re.DOTALL)
    entities = [p.strip() for p in re.split(r"\s*,\s*", m_ent.group(1)) if p.strip()] if m_ent else []
    m_opt = re.search(r"옵션\s*:\s*\{(.*?)\}", raw, flags=re.DOTALL)
    options = {"include_replies": False, "channel_filter": "any", "lang": "auto"}
    if m_opt:
        blob = m_opt.group(1)
        if ir := re.search(r"include_replies\s*:\s*(true|false)", blob, re.I): options["include_replies"] = (ir.group(1).lower()=="true")
        if cf := re.search(r"channel_filter\s*:\s*\"(any|official|unofficial)\"", blob, re.I): options["channel_filter"] = cf.group(1)
        if lg := re.search(r"lang\s*:\s*\"(ko|en|auto)\"", blob, re.I): options["lang"] = lg.group(1)
    if not start_iso or not end_iso:
        end_dt = now_kst(); start_dt = end_dt - timedelta(hours=24)
        start_iso, end_iso = to_iso_kst(start_dt), to_iso_kst(end_dt)
    if not keywords: keywords = [m[0]] if (m := re.findall(r"[가-힣A-Za-z0-9]{2,}", raw)) else ["유튜브"]
    return {"start_iso": start_iso, "end_iso": end_iso, "keywords": keywords, "entities": entities, "options": options, "raw": raw}

def yt_search_videos(rt, keyword, max_results, order="relevance", published_after=None, published_before=None):
    video_ids, token = [], None
    while len(video_ids) < max_results:
        params = dict(q=keyword, part="id", type="video", order=order, maxResults=min(50, max_results - len(video_ids)))
        if published_after: params["publishedAfter"]  = published_after
        if published_before: params["publishedBefore"] = published_before
        if token: params["pageToken"] = token
        resp = rt.execute(lambda s: s.search().list(**params))
        video_ids.extend(it["id"]["videoId"] for it in resp.get("items", []) if it["id"]["videoId"] not in video_ids)
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
            stats, snip, cont = item.get("statistics", {}), item.get("snippet", {}), item.get("contentDetails", {})
            dur = cont.get("duration","")
            h, m, s = re.search(r"(\d+)H", dur), re.search(r"(\d+)M", dur), re.search(r"(\d+)S", dur)
            dur_sec = (int(h.group(1)) * 3600 if h else 0) + (int(m.group(1)) * 60 if m else 0) + (int(s.group(1)) if s else 0)
            rows.append({"video_id": item.get("id"), "video_url": f"https://www.youtube.com/watch?v={item.get('id')}", "title": snip.get("title",""), "channelTitle": snip.get("channelTitle",""), "publishedAt": snip.get("publishedAt",""), "duration": dur, "shortType": "Shorts" if dur_sec <= 60 else "Clip", "viewCount": int(stats.get("viewCount",0) or 0), "likeCount": int(stats.get("likeCount",0) or 0), "commentCount": int(stats.get("commentCount",0) or 0)})
        time.sleep(0.25)
    return rows

def yt_all_replies(rt, parent_id, video_id, title="", short_type="Clip", cap=None):
    replies, token = [], None
    while not (cap is not None and len(replies) >= cap):
        try: resp = rt.execute(lambda s: s.comments().list(part="snippet", parentId=parent_id, maxResults=100, pageToken=token, textFormat="plainText"))
        except HttpError: break
        for c in resp.get("items", []):
            sn = c["snippet"]
            replies.append({"video_id": video_id, "video_title": title, "shortType": short_type, "comment_id": c.get("id",""), "parent_id": parent_id, "isReply": 1, "author": sn.get("authorDisplayName",""), "text": sn.get("textDisplay","") or "", "publishedAt": sn.get("publishedAt",""), "likeCount": int(sn.get("likeCount",0) or 0)})
        token = resp.get("nextPageToken")
        if not token: break
        time.sleep(0.2)
    return replies[:cap] if cap is not None else replies

def yt_all_comments_sync(rt, video_id, title="", short_type="Clip", include_replies=True, max_per_video=None):
    rows, token = [], None
    while not (max_per_video is not None and len(rows) >= max_per_video):
        try: resp = rt.execute(lambda s: s.commentThreads().list(part="snippet,replies", videoId=video_id, maxResults=100, pageToken=token, textFormat="plainText"))
        except HttpError: break
        for it in resp.get("items", []):
            top = it["snippet"]["topLevelComment"]["snippet"]
            thread_id = it["snippet"]["topLevelComment"]["id"]
            rows.append({"video_id": video_id, "video_title": title, "shortType": short_type, "comment_id": thread_id, "parent_id": "", "isReply": 0, "author": top.get("authorDisplayName",""), "text": top.get("textDisplay","") or "", "publishedAt": top.get("publishedAt",""), "likeCount": int(top.get("likeCount",0) or 0)})
            if include_replies and int(it["snippet"].get("totalReplyCount",0) or 0) > 0:
                cap = None if max_per_video is None else max(0, max_per_video - len(rows))
                if cap == 0: break
                rows.extend(yt_all_replies(rt, thread_id, video_id, title, short_type, cap=cap))
        token = resp.get("nextPageToken")
        if not token: break
        time.sleep(0.2)
    return rows[:max_per_video] if max_per_video is not None else rows

def parallel_collect_comments_streaming(video_list, rt_keys, include_replies, max_total_comments, max_per_video, prog_bar):
    out_csv = os.path.join(BASE_DIR, f"collect_{uuid4().hex}.csv")
    wrote_header = False; total_written = 0
    total_videos = len(video_list); done = 0
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(yt_all_comments_sync, RotatingYouTube(rt_keys), v["video_id"], v.get("title",""), v.get("shortType","Clip"), include_replies, max_per_video): v for v in video_list}
        for f in as_completed(futures):
            try:
                if comm := f.result():
                    dfc = pd.DataFrame(comm)
                    dfc.to_csv(out_csv, index=False, mode="a" if wrote_header else "w", header=not wrote_header, encoding="utf-8-sig")
                    wrote_header = True; total_written += len(dfc)
            except Exception: pass
            done += 1
            frac = 0.50 + (done / total_videos) * 0.40
            prog_bar.progress(min(0.90, frac), text="댓글 수집중…")
            if total_written >= max_total_comments: break
    return out_csv, total_written

def serialize_comments_for_llm_from_file(csv_path: str, max_chars_per_comment=280, max_total_chars=420_000):
    if not os.path.exists(csv_path): return "", 0, 0
    try: df_all = pd.read_csv(csv_path)
    except Exception: return "", 0, 0
    if df_all.empty: return "", 0, 0
    df_top_likes = df_all.sort_values("likeCount", ascending=False).head(1000)
    df_remaining = df_all.drop(df_top_likes.index)
    sample_size = min(1000, len(df_remaining))
    df_random = df_remaining.sample(n=sample_size) if sample_size > 0 else pd.DataFrame()
    df_sample = pd.concat([df_top_likes, df_random])
    lines, total_chars = [], 0
    for _, r in df_sample.iterrows():
        if total_chars >= max_total_chars: break
        text = str(r.get("text","") or "").replace("\n"," ")
        line = f"[{'R' if int(r.get('isReply',0))==1 else 'T'}|♥{int(r.get('likeCount',0))}] {str(r.get('author','')).replace('\n',' ')}: {text[:max_chars_per_comment] + '…' if len(text) > max_chars_per_comment else text}"
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
        cleaned.append(l)
        prev_blank = is_blank
    return "\n".join(cleaned).strip()

def run_pipeline_first_turn(user_query: str, prog_bar):
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
        for e in kw_ent:
            all_ids.extend(yt_search_videos(rt, f"{base_kw} {e}", 30, "relevance", kst_to_rfc3339_utc(start_dt), kst_to_rfc3339_utc(end_dt)))
    all_ids = list(dict.fromkeys(all_ids))
    prog_bar.progress(0.40, text="댓글 수집 준비중…")
    df_stats = pd.DataFrame(yt_video_statistics(rt, all_ids))
    st.session_state["last_df"] = df_stats
    csv_path, total_cnt = parallel_collect_comments_streaming(df_stats.to_dict('records'), YT_API_KEYS, bool(schema.get("options",{}).get("include_replies")), MAX_TOTAL_COMMENTS, MAX_COMMENTS_PER_VID, prog_bar)
    st.session_state["last_csv"] = csv_path
    if total_cnt == 0: return "지정 기간/키워드에서 댓글을 찾을 수 없습니다. 다른 조건으로 시도해 보세요."
    prog_bar.progress(0.90, text="AI 분석중…")
    sample_text, _, _ = serialize_comments_for_llm_from_file(csv_path)
    st.session_state["sample_text"] = sample_text
    sys = "너는 유튜브 댓글을 분석하는 어시스턴트다. 주어진 댓글 샘플을 바탕으로 핵심 포인트를 항목화하고, 긍/부/중 비율과 대표 코멘트(10개 미만)를 제시하라."
    payload = f"[키워드]: {', '.join(kw_main)}\n[엔티티]: {', '.join(kw_ent)}\n[기간(KST)]: {schema['start_iso']} ~ {schema['end_iso']}\n\n[댓글 샘플]:\n{sample_text}\n"
    answer_md_raw = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, sys, payload)
    prog_bar.progress(1.0, text="완료")
    time.sleep(0.5)
    return tidy_answer(answer_md_raw)

def run_followup_turn(user_query: str):
    if not (schema := st.session_state.get("last_schema")): return "오류: 이전 분석 기록이 없습니다. 새 채팅을 시작해주세요."
    sample_text = st.session_state.get("sample_text","")
    context = "\n".join(f"[이전 {'Q' if m['role']=='user' else 'A'}]: {m['content']}" for m in st.session_state["chat"][-10:])
    sys = "너는 유튜브 댓글 분석가다. 주어진 댓글 샘플과 이전 대화 맥락을 바탕으로 현재 질문에 간결하게 답하라. 반드시 댓글 샘플을 근거로 답하고, 인용은 5개 이하로 하라."
    payload = f"{context}\n\n[현재 질문]: {user_query}\n[기간(KST)]: {schema.get('start_iso','?')} ~ {schema.get('end_iso','?')}\n\n[댓글 샘플]:\n{sample_text}\n"
    return tidy_answer(call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, sys, payload))

# -------------------- 메인 화면 및 실행 로직 [전체 수정] --------------------

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
    render_metadata_outside_chat()
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    scroll_to_bottom()


if prompt := st.chat_input("예) 최근 24시간 태풍상사 김준호 반응 요약해줘"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.chat and st.session_state.chat[-1]["role"] == "user":
    user_query = st.session_state.chat[-1]["content"]
    
    with st.chat_message("assistant"):
        container = st.empty()
        
        if not st.session_state.get("last_csv"):
            progress_bar = container.progress(0, text="준비 중…")
            response = run_pipeline_first_turn(user_query, progress_bar)
        else:
            with container.spinner("💬 AI가 답변을 구성 중입니다..."):
                response = run_followup_turn(user_query)
        
        container.markdown(response)

    st.session_state.chat.append({"role": "assistant", "content": response})
    st.rerun()
