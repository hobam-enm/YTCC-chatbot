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
# 사이드바 열림으로 고정 요청 반영 (initial_sidebar_state="expanded")
st.set_page_config(page_title="유튜브 댓글분석: 챗봇", layout="wide", initial_sidebar_state="expanded")

# 챗봇 UI 느낌을 위해 제목 제거 및 페이지 상하좌우 패딩 최소화 CSS 주입 (요청하신 UI는 유지)
st.markdown("""
<style>
/* Streamlit 메인 컨테이너 패딩 최소화 */
.main .block-container {
    padding-top: 2rem;
    padding-right: 1rem;
    padding-left: 1rem;
    padding-bottom: 10rem;
}
/* 채팅 입력창이 고정될 수 있도록 여백 조정 */
[data-testid="stSidebarContent"] {
    padding-top: 1.5rem;
}
/* Streamlit 기본 헤더 및 푸터 숨기기 (더 깔끔한 UI를 위해) */
header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
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
        chat=[],                 # [{role, content}]  (content: markdown)
        meta_shown=False,        # 메타(키워드/기간) 표시했는지 여부 (첫 답변에만) - 사용하지 않음
        last_schema=None,        # dict
        last_csv="",             # csv path
        last_df=None,            # videos df
        last_keywords=[],        # list[str]
        last_entities=[],        # list[str]
        last_period=("", ""),    # (start_iso, end_iso)
        sample_text="",          # LLM sample text
    )
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
ensure_state()

# -------------------- [UI수정] 사이드바 --------------------
with st.sidebar:
    # CSS를 주입하여 '새 채팅' 버튼을 상단에, '문의' 정보를 하단에 고정
    st.markdown("""
    <style>
        [data-testid="stSidebarUserContent"] {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 4rem); /* 전체 뷰포트 높이에서 상단 패딩 제외 */
        }
        .contact-info {
            margin-top: auto; /* 이 요소가 남은 공간을 모두 차지하여 맨 아래로 밀려남 */
        }
    </style>
    """, unsafe_allow_html=True)

    # 3. '새 채팅' 버튼 (세련된 디자인, 최상단 배치)
    if st.button("✨ 새 채팅", use_container_width=True, type="secondary"):
        st.session_state.clear()
        fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
        if callable(fn): fn()

    # 2. 사이드바 정리: 다운로드 버튼 등 제거됨

    # 4. 문의처 (최하단 고정)
    st.markdown("""
    <div class="contact-info">
        <hr>
        <h3>📞 문의</h3>
        <p>미디어)디지털마케팅 데이터파트 김호범</p>
    </div>
    """, unsafe_allow_html=True)


# -------------------- 로직 (수정 없음) --------------------
def safe_rerun():
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(fn): fn()

def scroll_to_bottom():
    st_html("<script>window.scrollTo(0, document.body.scrollHeight);</script>", height=0)

def render_metadata_outside_chat():
    if not st.session_state.get("last_schema"): return

    schema = st.session_state["last_schema"]
    kw_main  = schema.get("keywords", [])
    start_iso = schema['start_iso']
    end_iso = schema['end_iso']

    try:
        start_dt_str = datetime.fromisoformat(start_iso).astimezone(KST).strftime('%Y-%m-%d %H:%M')
        end_dt_str = datetime.fromisoformat(end_iso).astimezone(KST).strftime('%Y-%m-%d %H:%M')
    except ValueError:
        start_dt_str = start_iso.split('T')[0]
        end_dt_str = end_iso.split('T')[0]

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
    keywords = []
    if m_kw:
        for part in re.split(r"\s*,\s*", m_kw.group(1)):
            p = re.sub(r"\(.*?\)", "", part).strip()
            if p: keywords.append(p)

    m_ent = re.search(r"엔티티/보조\s*:\s*\[(.*?)\]", raw, flags=re.DOTALL)
    entities = []
    if m_ent:
        for part in re.split(r"\s*,\s*", m_ent.group(1)):
            p = part.strip()
            if p: entities.append(p)

    m_opt = re.search(r"옵션\s*:\s*\{(.*?)\}", raw, flags=re.DOTALL)
    options = {"include_replies": False, "channel_filter": "any", "lang": "auto"}
    if m_opt:
        blob = m_opt.group(1)
        ir = re.search(r"include_replies\s*:\s*(true|false)", blob, re.I)
        cf = re.search(r"channel_filter\s*:\s*\"(any|official|unofficial)\"", blob, re.I)
        lg = re.search(r"lang\s*:\s*\"(ko|en|auto)\"", blob, re.I)
        if ir: options["include_replies"] = (ir.group(1).lower()=="true")
        if cf: options["channel_filter"] = cf.group(1)
        if lg: options["lang"] = lg.group(1)

    if not start_iso or not end_iso:
        end_dt = now_kst(); start_dt = end_dt - timedelta(hours=24)
        start_iso, end_iso = to_iso_kst(start_dt), to_iso_kst(end_dt)
    if not keywords:
        m = re.findall(r"[가-힣A-Za-z0-9]{2,}", raw)
        keywords = [m[0]] if m else ["유튜브"]

    return {"start_iso": start_iso, "end_iso": end_iso, "keywords": keywords, "entities": entities, "options": options, "raw": raw}

def yt_search_videos(rt, keyword, max_results, order="relevance", published_after=None, published_before=None):
    video_ids, token = [], None
    while len(video_ids) < max_results:
        params = dict(q=keyword, part="id", type="video", order=order, maxResults=min(50, max_results - len(video_ids)))
        if published_after: params["publishedAfter"]  = published_after
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
            dur = cont.get("duration","")
            def _dsec(d: str):
                if not d or not d.startswith("P"): return None
                h = re.search(r"(\d+)H", d); m = re.search(r"(\d+)M", d); s = re.search(r"(\d+)S", d)
                return (int(h.group(1)) if h else 0)*3600 + (int(m.group(1)) if m else 0)*60 + (int(s.group(1)) if s else 0)
            dur_sec = _dsec(dur)
            short_type = "Shorts" if (dur_sec is not None and dur_sec <= 60) else "Clip"
            vid = item.get("id")
            rows.append({
                "video_id": vid,
                "video_url": f"https://www.youtube.com/watch?v={vid}",
                "title": snip.get("title",""),
                "channelTitle": snip.get("channelTitle",""),
                "publishedAt": snip.get("publishedAt",""),
                "duration": dur,
                "shortType": short_type,
                "viewCount": int(stats.get("viewCount",0) or 0),
                "likeCount": int(stats.get("likeCount",0) or 0),
                "commentCount": int(stats.get("commentCount",0) or 0),
            })
        time.sleep(0.25)
    return rows

def yt_all_replies(rt, parent_id, video_id, title="", short_type="Clip", cap=None):
    replies, token = [], None
    while True:
        if cap is not None and len(replies) >= cap: return replies[:cap]
        params = dict(part="snippet", parentId=parent_id, maxResults=100, pageToken=token, textFormat="plainText")
        try:
            resp = rt.execute(lambda s: s.comments().list(**params))
        except HttpError:
            break
        for c in resp.get("items", []):
            sn = c["snippet"]
            replies.append({
                "video_id": video_id, "video_title": title, "shortType": short_type,
                "comment_id": c.get("id",""), "parent_id": parent_id, "isReply": 1,
                "author": sn.get("authorDisplayName",""),
                "text": sn.get("textDisplay","") or "",
                "publishedAt": sn.get("publishedAt",""),
                "likeCount": int(sn.get("likeCount",0) or 0),
            })
            if cap is not None and len(replies) >= cap: return replies[:cap]
        token = resp.get("nextPageToken")
        if not token: break
        time.sleep(0.2)
    return replies

def yt_all_comments_sync(rt, video_id, title="", short_type="Clip", include_replies=True, max_per_video=None):
    rows, token = [], None
    while True:
        if max_per_video is not None and len(rows) >= max_per_video: return rows[:max_per_video]
        params = dict(part="snippet,replies", videoId=video_id, maxResults=100, pageToken=token, textFormat="plainText")
        try:
            resp = rt.execute(lambda s: s.commentThreads().list(**params))
        except HttpError:
            break
        for it in resp.get("items", []):
            top = it["snippet"]["topLevelComment"]["snippet"]
            thread_id = it["snippet"]["topLevelComment"]["id"]
            total_replies = int(it["snippet"].get("totalReplyCount",0) or 0)
            rows.append({
                "video_id": video_id, "video_title": title, "shortType": short_type,
                "comment_id": thread_id, "parent_id": "", "isReply": 0,
                "author": top.get("authorDisplayName",""),
                "text": top.get("textDisplay","") or "",
                "publishedAt": top.get("publishedAt",""),
                "likeCount": int(top.get("likeCount",0) or 0),
            })
            if include_replies and total_replies>0:
                cap = None if max_per_video is None else max(0, max_per_video - len(rows))
                if cap == 0: return rows[:max_per_video]
                rows.extend(yt_all_replies(rt, thread_id, video_id, title, short_type, cap=cap))
                if max_per_video is not None and len(rows) >= max_per_video: return rows[:max_per_video]
        token = resp.get("nextPageToken")
        if not token: break
        time.sleep(0.2)
    return rows

def parallel_collect_comments_streaming(video_list, rt_keys, include_replies, max_total_comments, max_per_video, prog=None):
    out_csv = os.path.join(BASE_DIR, f"collect_{uuid4().hex}.csv")
    wrote_header = False; total_written = 0
    total_videos = len(video_list); done = 0
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(yt_all_comments_sync, RotatingYouTube(rt_keys), v["video_id"], v.get("title",""), v.get("shortType","Clip"), include_replies, max_per_video): v for v in video_list}
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
            if prog:
                frac = 0.50 + (done/total_videos) * 0.40
                prog.progress(min(0.90, frac), text="댓글 수집중…")
            if total_written >= max_total_comments: break
    return out_csv, total_written

def serialize_comments_for_llm_from_file(csv_path: str, max_rows=1500, max_chars_per_comment=280, max_total_chars=420_000):
    if not csv_path or not os.path.exists(csv_path): return "", 0, 0
    lines, total = [], 0; remaining = max_rows
    for chunk in pd.read_csv(csv_path, chunksize=120_000):
        if "likeCount" in chunk.columns: chunk = chunk.sort_values("likeCount", ascending=False)
        for _, r in chunk.iterrows():
            if remaining <= 0 or total >= max_total_chars: break
            is_reply = "R" if int(r.get("isReply",0) or 0)==1 else "T"
            author = str(r.get("author","") or "").replace("\n"," ")
            likec = int(r.get("likeCount",0) or 0)
            text = str(r.get("text","") or "").replace("\n"," ")
            if len(text) > max_chars_per_comment: text = text[:max_chars_per_comment] + "…"
            line = f"[{is_reply}|♥{likec}] {author}: {text}"
            if total + len(line) + 1 > max_total_chars: break
            lines.append(line); total += len(line)+1; remaining -= 1
        if remaining <= 0 or total >= max_total_chars: break
    return "\n".join(lines), len(lines), total

TITLE_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s+.*$")
HEADER_DUP_RE = re.compile(r"유튜브\s*댓글\s*분석.*", re.IGNORECASE)

def tidy_answer(md: str) -> str:
    if not md: return md
    lines = []
    for line in md.splitlines():
        if TITLE_LINE_RE.match(line):
            continue
        if HEADER_DUP_RE.search(line):
            continue
        lines.append(line)
    cleaned = []
    prev_blank = False
    for l in lines:
        if l.strip() == "":
            if prev_blank: continue
            prev_blank = True
        else:
            prev_blank = False
        cleaned.append(l)
    return "\n".join(cleaned).strip()

def render_chat():
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

def run_pipeline_first_turn(user_query: str):
    prog = st.progress(0.0, text="해석중…")
    if not GEMINI_API_KEYS:
        with st.chat_message("assistant"): st.markdown("Gemini API Key가 비어 있어요.")
        prog.progress(1.0, text="완료");
        return
    light = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, "", LIGHT_PROMPT.replace("{USER_QUERY}", user_query),
                                 timeout_s=GEMINI_TIMEOUT, max_tokens=GEMINI_MAX_TOKENS)
    schema = parse_light_block_to_schema(light)
    prog.progress(0.10, text="영상 수집중…")

    if not YT_API_KEYS:
        with st.chat_message("assistant"): st.markdown("YouTube API Key가 비어 있어요.")
        prog.progress(1.0, text="완료");
        return
    start_dt = datetime.fromisoformat(schema["start_iso"]).astimezone(KST)
    end_dt   = datetime.fromisoformat(schema["end_iso"]).astimezone(KST)
    published_after  = kst_to_rfc3339_utc(start_dt)
    published_before = kst_to_rfc3339_utc(end_dt)
    kw_main  = schema.get("keywords", [])
    kw_ent   = schema.get("entities", [])
    include_replies = bool(schema.get("options",{}).get("include_replies", False))

    rt = RotatingYouTube(YT_API_KEYS)
    all_ids = []
    for base_kw in (kw_main or ["유튜브"]):
        all_ids += yt_search_videos(rt, base_kw, 60, "relevance", published_after, published_before)
        for e in (kw_ent or []):
            all_ids += yt_search_videos(rt, f"{base_kw} {e}", 30, "relevance", published_after, published_before)
    all_ids = list(dict.fromkeys(all_ids))
    prog.progress(0.40, text="댓글 수집중…")

    stats = yt_video_statistics(rt, all_ids)
    df_stats = pd.DataFrame(stats)
    csv_path, total_cnt = parallel_collect_comments_streaming(
        video_list=df_stats.to_dict('records'),
        rt_keys=YT_API_KEYS,
        include_replies=include_replies,
        max_total_comments=MAX_TOTAL_COMMENTS,
        max_per_video=MAX_COMMENTS_PER_VID,
        prog=prog,
    )
    if total_cnt == 0:
        prog.progress(1.0, text="완료")
        with st.chat_message("assistant"):
            st.markdown("지정 기간/키워드에서 댓글이 보이지 않아. 기간/키워드를 조정해줘.")
        st.session_state["chat"].append({"role":"assistant","content":"지정 기간/키워드에서 댓글이 보이지 않아. 기간/키워드를 조정해줘."})
        scroll_to_bottom()
        return

    prog.progress(0.90, text="AI 분석중…")
    sample_text, _, _ = serialize_comments_for_llm_from_file(csv_path)
    sys = ("너는 유튜브 댓글을 분석하는 어시스턴트다. "
           "아래 키워드/엔티티와 지정된 기간의 댓글 샘플을 바탕으로 핵심 포인트를 항목화하고, "
           "긍/부/중 비율과 대표 코멘트(10개 미만)를 제시하라.")
    payload = (
        f"[키워드]: {', '.join(kw_main)}\n"
        f"[엔티티]: {', '.join(kw_ent)}\n"
        f"[기간(KST)]: {schema['start_iso']} ~ {schema['end_iso']}\n\n"
        f"[댓글 샘플]:\n{sample_text}\n"
    )
    answer_md_raw = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, sys, payload,
                                         timeout_s=GEMINI_TIMEOUT, max_tokens=GEMINI_MAX_TOKENS)
    answer_md = tidy_answer(answer_md_raw)
    prog.progress(1.0, text="완료")

    st.session_state["last_schema"]   = schema
    st.session_state["last_csv"]      = csv_path
    st.session_state["last_df"]       = df_stats
    st.session_state["sample_text"]   = sample_text
    st.session_state["last_keywords"] = kw_main
    st.session_state["last_entities"] = kw_ent
    st.session_state["last_period"]   = (schema["start_iso"], schema["end_iso"])

    with st.chat_message("assistant"):
        st.markdown(answer_md)
    st.session_state["chat"].append({"role":"assistant","content": answer_md})

    scroll_to_bottom()
    safe_rerun()

def run_followup_turn(user_query: str):
    schema = st.session_state.get("last_schema") or {}
    sample_text = st.session_state.get("sample_text","")

    lines = []
    for m in st.session_state["chat"][-10:]:
        content_text = re.sub(r'<div.*?/div>', '', m['content'], flags=re.DOTALL).strip()
        if m["role"] == "user": lines.append(f"[이전 Q]: {content_text}")
        else:                   lines.append(f"[이전 A]: {content_text}")
    context = "\n".join(lines)

    sys = ("너는 유튜브 댓글을 분석하는 어시스턴트다. "
           "아래는 직렬화된 댓글 샘플(고정)과 이전 대화 맥락이다. "
           "현재 질문에 대해 간결하고 구조화된 답을 한국어로 하라. "
           "반드시 댓글 샘플을 근거로 답하고, 인용 예시는 5개 이하로 제시하라.")
    payload = (
        context + "\n\n" +
        f"[현재 질문]: {user_query}\n"
        f"[기간(KST)]: {schema.get('start_iso','?')} ~ {schema.get('end_iso','?')}\n\n"
        f"[댓글 샘플]:\n{sample_text}\n"
    )

    with st.spinner("💬 AI가 답변을 구성 중입니다..."):
        answer_md_raw = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, sys, payload,
                                             timeout_s=GEMINI_TIMEOUT, max_tokens=GEMINI_MAX_TOKENS)
    
    answer_md = tidy_answer(answer_md_raw)

    with st.chat_message("assistant"):
        st.markdown(answer_md)
    st.session_state["chat"].append({"role":"assistant","content": answer_md})
    scroll_to_bottom()

# -------------------- [UI수정] 메인 화면 --------------------

# 5. 초기 화면과 채팅 화면 분리 (Gemini 스타일)
if not st.session_state["chat"]:
    # 1. 제미나이 스타일 초기 화면 구성
    st.markdown("""
        <style>
            .welcome-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                height: 70vh;
            }
            .welcome-container h1 {
                font-size: 3.5rem;
                font-weight: 600;
                background: -webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570, #F2A60C);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .welcome-container .subtitle {
                font-size: 1.2rem;
                color: #4b5563;
            }
            .usage-notice {
                margin-top: 3rem;
                padding: 1rem 1.5rem;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                background-color: #fafafa;
                max-width: 600px;
            }
             .usage-notice h4 {
                margin-bottom: 1rem;
                font-weight: 600;
             }
        </style>
        <div class="welcome-container">
            <h1>유튜브 댓글분석: AI 챗봇</h1>
            <p class="subtitle">기간과 분석주제를 명시하여 대화를 시작하세요</p>
            <div class="usage-notice">
                <h4>⚠️ 사용 주의사항</h4>
                <ol style="text-align: left; padding-left: 20px;">
                    <li><strong>첫 질문 시</strong> 댓글 수집 및 AI 분석에 다소 시간이 소요될 수 있습니다.</li>
                    <li>한 세션에서는 <strong>하나의 주제</strong>와 관련된 질문만 진행해야 분석 정확도가 유지됩니다.</li>
                </ol>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    # 채팅이 시작되면 기존 채팅 화면을 렌더링
    render_metadata_outside_chat()
    render_chat()

# 채팅 입력창 (Streamlit 기본 기능으로 하단에 고정됨)
prompt = st.chat_input(placeholder="예) 최근 24시간 태풍상사 김준호 반응 요약해줘")
if prompt:
    # 채팅 시작 시, 이전의 초기화면 안내 메시지 등은 rerun 시 사라짐
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.session_state["chat"].append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # safe_rerun() 대신 scroll_to_bottom() 호출하여 즉시 스크롤
    scroll_to_bottom()

    # 파이프라인 실행
    if st.session_state.get("last_csv"):
        run_followup_turn(prompt)
    else:
        run_pipeline_first_turn(prompt)
