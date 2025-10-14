# -*- coding: utf-8 -*-
# 💬 유튜브 댓글분석기 — 챗봇 모드 (독립 앱)
# - 자연어 한 줄 → (기간/키워드/옵션) 해석 → 영상 수집 → 댓글 수집(스트리밍) → 요약/시각화
# - 해석은 자유형(제미나이), 어댑터에서만 규격화(KST ISO, 키워드 리스트 등)
# - Streamlit Cloud 기준 /tmp 사용, GitHub 아카이브 옵션 제외(심플)

import streamlit as st
import pandas as pd
import os, re, io, gc, time, base64, json, requests
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from uuid import uuid4

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import google.generativeai as genai

import plotly.express as px
from plotly import graph_objects as go
import circlify
import numpy as np

# =====================================================
# 기본 설정
# =====================================================
st.set_page_config(page_title="💬 유튜브 댓글분석기: 챗봇 모드", layout="wide", initial_sidebar_state="collapsed")
st.title("💬 유튜브 댓글분석기: 챗봇 모드 (베타)")

# ===================== 경로/상수 =====================
BASE_DIR = "/tmp"
os.makedirs(BASE_DIR, exist_ok=True)

MAX_TOTAL_COMMENTS = 120_000
MAX_COMMENTS_PER_VIDEO = 4_000
GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_TIMEOUT = int(st.secrets.get("GEMINI_TIMEOUT", 120))
GEMINI_MAX_TOKENS = int(st.secrets.get("GEMINI_MAX_TOKENS", 2048))

# ===================== 비밀키 =====================
_YT_FALLBACK = []
_GEM_FALLBACK = []
YT_API_KEYS = list(st.secrets.get("YT_API_KEYS", [])) or _YT_FALLBACK
GEMINI_API_KEYS = list(st.secrets.get("GEMINI_API_KEYS", [])) or _GEM_FALLBACK

# ===================== 유틸/공통 =====================
KST = timezone(timedelta(hours=9))

def now_kst() -> datetime:
    return datetime.now(tz=KST)

def to_iso_kst(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST).isoformat(timespec="seconds")

def kst_to_rfc3339_utc(dt_kst: datetime) -> str:
    if dt_kst.tzinfo is None:
        dt_kst = dt_kst.replace(tzinfo=KST)
    return dt_kst.astimezone(timezone.utc).isoformat().replace("+00:00","Z")

# ===================== Streamlit rerun 호환 =====================
def safe_rerun():
    fn = getattr(st, "rerun", None)
    if callable(fn):
        return fn()
    fn_old = getattr(st, "experimental_rerun", None)
    if callable(fn_old):
        return fn_old()
    raise RuntimeError("No rerun function available.")

# ===================== 키 로테이터 =====================
class RotatingKeys:
    def __init__(self, keys, state_key: str, on_rotate=None, treat_as_strings: bool = True):
        cleaned = []
        for k in (keys or []):
            if k is None: continue
            if treat_as_strings and isinstance(k, str):
                ks = k.strip()
                if ks: cleaned.append(ks)
            else:
                cleaned.append(k)
        self.keys = cleaned[:10]
        self.state_key = state_key
        self.on_rotate = on_rotate
        idx = st.session_state.get(state_key, 0)
        self.idx = 0 if not self.keys else (idx % len(self.keys))
        st.session_state[state_key] = self.idx
    def current(self):
        if not self.keys: return None
        return self.keys[self.idx % len(self.keys)]
    def rotate(self):
        if not self.keys: return
        self.idx = (self.idx + 1) % len(self.keys)
        st.session_state[self.state_key] = self.idx
        if callable(self.on_rotate): self.on_rotate(self.idx, self.current())

# ===================== YouTube 래퍼 =====================
class RotatingYouTube:
    def __init__(self, keys, state_key="yt_key_idx", log=None):
        self.rot = RotatingKeys(keys, state_key, on_rotate=lambda i, k: log and log(f"🔁 YouTube 키 전환 → #{i+1}"))
        self.log = log
        self.service = None
        self._build_service()
    def _build_service(self):
        key = self.rot.current()
        if not key:
            raise RuntimeError("YouTube API Key가 비어 있습니다.")
        self.service = build("youtube", "v3", developerKey=key)
    def _rotate_and_rebuild(self):
        self.rot.rotate(); self._build_service()
    def execute(self, request_factory, tries_per_key=2):
        attempts = 0
        max_attempts = len(self.rot.keys) if self.rot.keys else 1
        while attempts < max_attempts:
            try:
                req = request_factory(self.service)
                return req.execute()
            except HttpError as e:
                status = getattr(getattr(e, 'resp', None), 'status', None)
                msg = (getattr(e, 'content', b'').decode('utf-8', errors='ignore') or '').lower()
                quotaish = status in (403,429) and (('quota' in msg) or ('rate' in msg) or ('limit' in msg))
                if quotaish and len(self.rot.keys) > 1:
                    self._rotate_and_rebuild(); attempts += 1; continue
                raise

# ===================== Gemini 호출 =====================
def is_gemini_quota_error(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return ("429" in msg) or ("too many requests" in msg) or ("rate limit" in msg) or ("resource exhausted" in msg) or ("quota" in msg)

def call_gemini_rotating(
    model_name: str,
    keys,
    system_instruction: str,
    user_payload: str,
    timeout_s: int = GEMINI_TIMEOUT,
    max_tokens: int = GEMINI_MAX_TOKENS,
    on_rotate=None
) -> str:
    rot = RotatingKeys(keys, state_key="gem_key_idx", on_rotate=lambda i, k: on_rotate and on_rotate(i, k))
    if not rot.current():
        raise RuntimeError("Gemini API Key가 비어 있습니다.")
    attempts = 0
    max_attempts = len(rot.keys) if rot.keys else 1
    while attempts < max_attempts:
        try:
            genai.configure(api_key=rot.current())
            model = genai.GenerativeModel(
                model_name,
                generation_config={"temperature": 0.2, "max_output_tokens": max_tokens, "top_p": 0.9}
            )
            resp = model.generate_content([system_instruction, user_payload], request_options={"timeout": timeout_s})
            out = getattr(resp, "text", None)
            if not out and hasattr(resp, "candidates") and resp.candidates:
                c0 = resp.candidates[0]
                if hasattr(c0, "content") and getattr(c0.content, "parts", None):
                    p0 = c0.content.parts[0]
                    if hasattr(p0, "text"):
                        out = p0.text
            return out or ""
        except Exception as e:
            if is_gemini_quota_error(e) and len(rot.keys) > 1:
                rot.rotate(); attempts += 1; continue
            raise

# ===================== 자연어 → 라이트 요약 블록 프롬프트 =====================
LIGHT_PROMPT = (
    "역할: 당신은 ‘유튜브 댓글 반응 분석기’를 위한 자연어 해석가다.\n"
    "목표: 사용자가 한국어로 말한 요청에서 [검색 기간]과 [검색 키워드(주제/엔티티/보조어)]를 최대한 정확히 해석한다.\n"
    "원칙:\n"
    "- 사용자의 표현을 존중한다. 자의적 축약·삭제 금지.\n"
    "- 기간은 한국 표준시(Asia/Seoul, +09:00) 기준으로 해석한다.\n"
    "- ‘최근 N시간/일/주/개월/년’ 같은 상대 기간은 종료시점을 ‘지금’으로 본다.\n"
    "- 절대 기간(예: 2025-09-01~2025-09-07, 어제 18시~오늘 9시)은 그대로 계산한다.\n"
    "- 작품/브랜드/사람 이름처럼 의미 있는 고유명사는 원문 표기를 보존한다.\n"
    "- 옵션이 자연어에 있으면 감지한다: 대댓글 포함/제외, 공식 채널만/비공식, 언어(한국어만/영어만/자동).\n\n"
    "출력 형식(사람이 읽기 쉬운 라이트 요약; 이 블록만 규칙적으로 써라):\n"
    "- 한 줄 요약: <한 문장으로 해석 결과 요약>\n"
    "- 기간(KST): <YYYY-MM-DDTHH:MM:SS+09:00> ~ <YYYY-MM-DDTHH:MM:SS+09:00>\n"
    "- 키워드: [<메인 키워드 1>, <메인 키워드 2> ...]\n"
    "- 엔티티/보조: [<인물/보조 키워드들, 없으면 빈 배열>]\n"
    "- 옵션: { include_replies: true|false, channel_filter: \"any|official|unofficial\", lang: \"ko|en|auto\" }\n"
    "- 원문: {USER_QUERY}\n\n"
    f"지금 시간은 KST 기준으로 \"{to_iso_kst(now_kst())}\" 이다.\n"
    "아래 사용자 입력을 해석하라:\n\n{USER_QUERY}"
)

# ===================== 라이트 블록 → 표준 스키마 어댑터 =====================
# 표준 스키마: {start_iso, end_iso, keywords[], entities[], options{}, raw}

def parse_light_block_to_schema(light_text: str) -> dict:
    raw = (light_text or "").strip()
    # 1) 각 라인 캡처
    # - 기간(KST): ... ~ ...
    m_time = re.search(r"기간\(KST\)\s*:\s*([^~]+)~\s*([^\n]+)", raw)
    start_iso = end_iso = None
    if m_time:
        start_iso = m_time.group(1).strip()
        end_iso = m_time.group(2).strip()
    # - 키워드: [ ... ]
    m_kw = re.search(r"키워드\s*:\s*\[(.*?)\]", raw, flags=re.DOTALL)
    keywords = []
    if m_kw:
        body = m_kw.group(1)
        # 항목은 쉼표로 분리, 괄호 후보는 제거하여 메인표기를 우선 보존
        for part in re.split(r"\s*,\s*", body):
            part = part.strip()
            if not part:
                continue
            # 괄호 내 후보(예: 태풍 상사(태풍상사)) → 바깥표기 우선
            part = re.sub(r"\(.*?\)", "", part).strip()
            if part:
                keywords.append(part)
    # - 엔티티/보조: [ ... ]
    m_ent = re.search(r"엔티티/보조\s*:\s*\[(.*?)\]", raw, flags=re.DOTALL)
    entities = []
    if m_ent:
        body = m_ent.group(1)
        for part in re.split(r"\s*,\s*", body):
            part = part.strip()
            if part:
                entities.append(part)
    # - 옵션: { ... }
    m_opt = re.search(r"옵션\s*:\s*\{(.*?)\}", raw, flags=re.DOTALL)
    options = {"include_replies": False, "channel_filter": "any", "lang": "auto"}
    if m_opt:
        blob = m_opt.group(1)
        inc = re.search(r"include_replies\s*:\s*(true|false)", blob, re.IGNORECASE)
        if inc:
            options["include_replies"] = (inc.group(1).lower() == "true")
        ch = re.search(r"channel_filter\s*:\s*\"(any|official|unofficial)\"", blob, re.IGNORECASE)
        if ch:
            options["channel_filter"] = ch.group(1)
        lg = re.search(r"lang\s*:\s*\"(ko|en|auto)\"", blob, re.IGNORECASE)
        if lg:
            options["lang"] = lg.group(1)
    # 안전 보정
    if not start_iso or not end_iso:
        # 상대기간 누락 등 → 기본 최근 24시간
        end_dt = now_kst(); start_dt = end_dt - timedelta(hours=24)
        start_iso, end_iso = to_iso_kst(start_dt), to_iso_kst(end_dt)
    # 키워드 비었을 때 안전값
    if not keywords:
        # 따옴표 안 최대 토큰 or 전체 문자열의 긴 한글 토큰 시도
        m = re.findall(r"[\"'“”‘’](.*?)[\"'“”‘’]", raw)
        if m:
            keywords = [s.strip() for s in m if s.strip()][:1]
        if not keywords:
            m2 = re.findall(r"[가-힣A-Za-z0-9]{2,}", raw)
            keywords = [m2[0]] if m2 else ["유튜브"]
    # 공백제거된 버전 보조(검색 정확도 향상), 다만 표준 스키마엔 원문형 보존
    return {
        "start_iso": start_iso,
        "end_iso": end_iso,
        "keywords": keywords,
        "entities": entities,
        "options": options,
        "raw": raw,
    }

# ===================== YouTube 검색/통계 =====================
_YT_ID_RE = re.compile(r'^[A-Za-z0-9_-]{11}$')

def yt_search_videos(rt, keyword, max_results, order="relevance", published_after=None, published_before=None, log=None):
    video_ids, token = [], None
    while len(video_ids) < max_results:
        params = dict(q=keyword, part="id", type="video", order=order, maxResults=min(50, max_results - len(video_ids)))
        if published_after: params["publishedAfter"] = published_after
        if published_before: params["publishedBefore"] = published_before
        if token: params["pageToken"] = token
        resp = rt.execute(lambda s: s.search().list(**params))
        for it in resp.get("items", []):
            vid = it["id"]["videoId"]
            if vid not in video_ids: video_ids.append(vid)
        token = resp.get("nextPageToken")
        if not token: break
        if log: log(f"검색 진행: {len(video_ids)}개")
        time.sleep(0.3)
    return video_ids

def yt_video_statistics(rt, video_ids, log=None):
    rows = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        if not batch: continue
        resp = rt.execute(lambda s: s.videos().list(part="statistics,snippet,contentDetails", id=",".join(batch)))
        for item in resp.get("items", []):
            stats = item.get("statistics", {})
            snip = item.get("snippet", {})
            cont = item.get("contentDetails", {})
            dur_iso = cont.get("duration", "")
            def _dsec(dur: str):
                if not dur or not dur.startswith("P"): return None
                h = re.search(r"(\d+)H", dur); m = re.search(r"(\d+)M", dur); s = re.search(r"(\d+)S", dur)
                return (int(h.group(1)) if h else 0) * 3600 + (int(m.group(1)) if m else 0) * 60 + (int(s.group(1)) if s else 0)
            dur_sec = _dsec(dur_iso)
            short_type = "Shorts" if (dur_sec is not None and dur_sec <= 60) else "Clip"
            vid_id = item.get("id")
            rows.append({
                "video_id": vid_id,
                "video_url": f"https://www.youtube.com/watch?v={vid_id}",
                "title": snip.get("title", ""),
                "channelTitle": snip.get("channelTitle", ""),
                "publishedAt": snip.get("publishedAt", ""),
                "duration": dur_iso,
                "shortType": short_type,
                "viewCount": int(stats.get("viewCount", 0) or 0),
                "likeCount": int(stats.get("likeCount", 0) or 0),
                "commentCount": int(stats.get("commentCount", 0) or 0),
            })
        if log: log(f"통계 배치 {i // 50 + 1} 완료")
        time.sleep(0.3)
    return rows

# ===================== 댓글 수집(스레드) + CSV 스트리밍 =====================

def yt_all_replies(rt, parent_id, video_id, title="", short_type="Clip", log=None, cap=None):
    replies, token = [], None
    while True:
        if cap is not None and len(replies) >= cap:
            return replies[:cap]
        params = dict(part="snippet", parentId=parent_id, maxResults=100, pageToken=token, textFormat="plainText")
        try:
            resp = rt.execute(lambda s: s.comments().list(**params))
        except HttpError as e:
            if log: log(f"[오류] replies {video_id}/{parent_id}: {e}")
            break
        for c in resp.get("items", []):
            sn = c["snippet"]
            replies.append({
                "video_id": video_id, "video_title": title, "shortType": short_type,
                "comment_id": c.get("id", ""), "parent_id": parent_id, "isReply": 1,
                "author": sn.get("authorDisplayName", ""),
                "text": sn.get("textDisplay", "") or "",
                "publishedAt": sn.get("publishedAt", ""),
                "likeCount": int(sn.get("likeCount", 0) or 0),
            })
            if cap is not None and len(replies) >= cap:
                return replies[:cap]
        token = resp.get("nextPageToken")
        if not token: break
        time.sleep(0.2)
    return replies


def yt_all_comments_sync(rt, video_id, title="", short_type="Clip", include_replies=True, log=None, max_per_video: int | None = None):
    rows, token = [], None
    while True:
        if max_per_video is not None and len(rows) >= max_per_video:
            return rows[:max_per_video]
        params = dict(part="snippet,replies", videoId=video_id, maxResults=100, pageToken=token, textFormat="plainText")
        try:
            resp = rt.execute(lambda s: s.commentThreads().list(**params))
        except HttpError as e:
            if log: log(f"[오류] commentThreads {video_id}: {e}")
            break
        for it in resp.get("items", []):
            top = it["snippet"]["topLevelComment"]["snippet"]
            thread_id = it["snippet"]["topLevelComment"]["id"]
            total_replies = int(it["snippet"].get("totalReplyCount", 0) or 0)
            rows.append({
                "video_id": video_id, "video_title": title, "shortType": short_type,
                "comment_id": thread_id, "parent_id": "", "isReply": 0,
                "author": top.get("authorDisplayName", ""),
                "text": top.get("textDisplay", "") or "",
                "publishedAt": top.get("publishedAt", ""),
                "likeCount": int(top.get("likeCount", 0) or 0),
            })
            if include_replies and total_replies > 0:
                cap = None
                if max_per_video is not None:
                    cap = max(0, max_per_video - len(rows))
                if cap == 0:
                    return rows[:max_per_video]
                rows.extend(yt_all_replies(rt, thread_id, video_id, title, short_type, log, cap=cap))
                if max_per_video is not None and len(rows) >= max_per_video:
                    return rows[:max_per_video]
        token = resp.get("nextPageToken")
        if not token: break
        if log: log(f"  댓글 페이지 진행, 누계 {len(rows)}")
        time.sleep(0.2)
    return rows


def parallel_collect_comments_streaming(video_list, rt_keys, include_replies, max_total_comments, max_per_video, log_callback=None, prog_callback=None):
    out_csv = os.path.join(BASE_DIR, f"collect_{uuid4().hex}.csv")
    wrote_header = False
    total_written = 0
    total_videos = len(video_list)
    done_videos = 0
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                yt_all_comments_sync,
                RotatingYouTube(rt_keys),
                vid_info["video_id"],
                vid_info.get("title", ""),
                vid_info.get("shortType", "Clip"),
                include_replies,
                None,
                max_per_video
            ): vid_info for vid_info in video_list
        }
        for fut in as_completed(futures):
            vid_info = futures[fut]
            try:
                comments = fut.result()
                if comments:
                    df_chunk = pd.DataFrame(comments)
                    df_chunk.to_csv(
                        out_csv, index=False,
                        mode=("a" if wrote_header else "w"),
                        header=(not wrote_header),
                        encoding="utf-8-sig"
                    )
                    wrote_header = True
                    total_written += len(df_chunk)
                done_videos += 1
                if log_callback: log_callback(f"✅ [{done_videos}/{total_videos}] {vid_info.get('title','')} - {len(comments):,}개 수집")
                if prog_callback: prog_callback(done_videos / total_videos)
            except Exception as e:
                done_videos += 1
                if log_callback: log_callback(f"❌ [{done_videos}/{total_videos}] {vid_info.get('title','')} - 실패: {e}")
                if prog_callback: prog_callback(done_videos / total_videos)
            if total_written >= max_total_comments:
                if log_callback: log_callback(f"최대 수집 한도({max_total_comments:,}개) 도달, 중단")
                break
    return out_csv, total_written

# ===================== LLM용 직렬화(샘플) =====================

def serialize_comments_for_llm_from_file(csv_path: str, max_rows=1500, max_chars_per_comment=280, max_total_chars=420_000):
    if not csv_path or not os.path.exists(csv_path):
        return "", 0, 0
    lines, total = [], 0
    remaining = max_rows
    for chunk in pd.read_csv(csv_path, chunksize=120_000):
        if "likeCount" in chunk.columns:
            chunk = chunk.sort_values("likeCount", ascending=False)
        for _, r in chunk.iterrows():
            if remaining <= 0 or total >= max_total_chars:
                break
            is_reply = "R" if int(r.get("isReply", 0) or 0) == 1 else "T"
            author = str(r.get("author", "") or "").replace("\n", " ")
            likec = int(r.get("likeCount", 0) or 0)
            text = str(r.get("text", "") or "").replace("\n", " ")
            if len(text) > max_chars_per_comment:
                text = text[:max_chars_per_comment] + "…"
            line = f"[{is_reply}|♥{likec}] {author}: {text}"
            if total + len(line) + 1 > max_total_chars:
                break
            lines.append(line)
            total += len(line) + 1
            remaining -= 1
        if remaining <= 0 or total >= max_total_chars:
            break
    return "\n".join(lines), len(lines), total

# ===================== 정량 시각화(간단판) =====================

def timeseries_from_file(csv_path: str):
    if not csv_path or not os.path.exists(csv_path): return None, None
    tmin = None; tmax = None
    for chunk in pd.read_csv(csv_path, usecols=["publishedAt"], chunksize=200_000):
        dt = pd.to_datetime(chunk["publishedAt"], errors="coerce", utc=True)
        if dt.notna().any():
            lo, hi = dt.min(), dt.max()
            tmin = lo if (tmin is None or (lo < tmin)) else tmin
            tmax = hi if (tmax is None or (hi > tmax)) else tmax
    if tmin is None or tmax is None:
        return None, None
    span_hours = (tmax - tmin).total_seconds()/3600.0
    use_hour = (span_hours <= 48)

    agg = {}
    for chunk in pd.read_csv(csv_path, usecols=["publishedAt"], chunksize=200_000):
        dt = pd.to_datetime(chunk["publishedAt"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")
        dt = dt.dropna()
        if dt.empty: continue
        bucket = (dt.dt.floor("H") if use_hour else dt.dt.floor("D"))
        vc = bucket.value_counts()
        for t, c in vc.items():
            agg[t] = agg.get(t, 0) + int(c)
    ts = pd.Series(agg).sort_index().rename("count").reset_index().rename(columns={"index":"bucket"})
    return ts, ("시간별" if use_hour else "일자별")

# ===================== UI — 입력/실행 =====================
with st.container(border=True):
    st.subheader("한 줄 요청")
    user_query = st.text_input(
        "챗봇에게 말하듯 입력하세요",
        placeholder="예) 최근 12시간 태풍상사 김준호 댓글반응 분석해줘",
        key="cb_query",
    )
    colA, colB = st.columns([1,1])
    btn_parse = colA.button("🧭 해석만", type="secondary")
    btn_run = colB.button("🚀 즉시 실행", type="primary")

# ===================== 해석 단계 =====================
light_block = None
schema = None
if btn_parse or btn_run:
    if not GEMINI_API_KEYS:
        st.error("Gemini API Key가 없습니다. st.secrets에 GEMINI_API_KEYS를 설정하세요.")
    else:
        with st.status("제미나이 해석 중…", expanded=True) as status:
            payload = LIGHT_PROMPT.replace("{USER_QUERY}", user_query or "").replace("{NOW_KST_ISO}", to_iso_kst(now_kst()))
            out = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, "", payload,
                                       timeout_s=GEMINI_TIMEOUT, max_tokens=GEMINI_MAX_TOKENS,
                                       on_rotate=lambda i, k: status.write(f"🔁 Gemini 키 전환 → #{i+1}"))
            light_block = out
            st.markdown("#### 🔎 라이트 요약 블록 (Gemini 원문)")
            st.code(light_block or "(빈 응답)")
            schema = parse_light_block_to_schema(light_block or "")
            st.markdown("#### 🧱 규격화 스키마")
            st.json(schema)
            status.update(label="해석 완료", state="complete")

# ===================== 실행(수집→요약) =====================
if btn_run and schema:
    if not YT_API_KEYS:
        st.error("YouTube API Key가 없습니다. st.secrets에 YT_API_KEYS를 설정하세요.")
    else:
        start_dt = datetime.fromisoformat(schema["start_iso"]).astimezone(KST)
        end_dt   = datetime.fromisoformat(schema["end_iso"]).astimezone(KST)
        kw_main  = schema.get("keywords", [])
        kw_entities = schema.get("entities", [])
        include_replies = bool(schema.get("options", {}).get("include_replies", False))

        # 검색 기간 RFC3339(UTC)
        published_after = kst_to_rfc3339_utc(start_dt)
        published_before = kst_to_rfc3339_utc(end_dt)

        with st.status("영상/댓글 수집 중…", expanded=True) as status:
            rt = RotatingYouTube(YT_API_KEYS, log=lambda m: status.write(m))
            all_ids = []
            # 메인 키워드 + 엔티티 조합으로 검색 폭을 넓힌 뒤 dedupe
            for base_kw in (kw_main or ["유튜브"]):
                ids = yt_search_videos(rt, base_kw, max_results=60, order="relevance",
                                       published_after=published_after, published_before=published_before,
                                       log=status.write)
                all_ids.extend(ids)
                # 엔티티 결합 쿼리도 시도
                for e in (kw_entities or []):
                    q2 = f"{base_kw} {e}"
                    ids2 = yt_search_videos(rt, q2, max_results=30, order="relevance",
                                            published_after=published_after, published_before=published_before,
                                            log=None)
                    all_ids.extend(ids2)
            all_ids = list(dict.fromkeys(all_ids))
            status.write(f"🎞️ 대상 영상: {len(all_ids)}개")

            stats = yt_video_statistics(rt, all_ids, log=status.write)
            df_stats = pd.DataFrame(stats)
            if not df_stats.empty and "publishedAt" in df_stats.columns:
                df_stats["publishedAt_kst"] = (
                    pd.to_datetime(df_stats["publishedAt"], errors="coerce", utc=True)
                    .dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M:%S")
                )
            st.dataframe(df_stats.head(20), use_container_width=True)

            status.write("💬 댓글 수집(스트리밍)…")
            video_list = df_stats.to_dict('records') if not df_stats.empty else []
            prog = st.progress(0, text="수집 진행 중")
            log_ph = st.empty()
            csv_path, total_cnt = parallel_collect_comments_streaming(
                video_list=video_list,
                rt_keys=YT_API_KEYS,
                include_replies=include_replies,
                max_total_comments=MAX_TOTAL_COMMENTS,
                max_per_video=MAX_COMMENTS_PER_VIDEO,
                log_callback=log_ph.write,
                prog_callback=prog.progress
            )
            status.write(f"총 댓글 수집: {total_cnt:,}개")
            status.update(label="수집 완료", state="complete")

        if total_cnt == 0:
            st.warning("수집된 댓글이 없습니다. 기간/키워드를 조정해 보세요.")
        else:
            # ===== AI 요약 =====
            st.markdown("---")
            st.subheader("🧠 AI 요약")
            a_text, _, _ = serialize_comments_for_llm_from_file(csv_path)
            system_instruction = (
                "너는 유튜브 댓글을 분석하는 어시스턴트다. "
                "아래 키워드/엔티티와 지정된 기간 내 댓글 샘플을 바탕으로, 전반적 반응을 한국어로 간결하게 요약하라. "
                "핵심 포인트를 항목화하고, 긍/부정/중립의 대략적 비율과 대표 코멘트(10개미만)를 예시로 제시하라. "
                "반드시 샘플을 근거로 작성하라."
            )
            prompt_q = (
                f"[키워드]: {', '.join(kw_main or [])}\n"
                f"[엔티티]: {', '.join(kw_entities or [])}\n"
                f"[기간(KST)]: {schema['start_iso']} ~ {schema['end_iso']}\n\n"
                f"[댓글 샘플]:\n{a_text}\n"
            )
            out = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, system_instruction, prompt_q,
                                       timeout_s=GEMINI_TIMEOUT, max_tokens=GEMINI_MAX_TOKENS)
            st.markdown(out)

            # ===== 정량 하이라이트 =====
            st.markdown("---")
            st.subheader("📊 정량 하이라이트")
            # 시점별 추이
            ts, label = timeseries_from_file(csv_path)
            if ts is not None and not ts.empty:
                fig_ts = px.line(ts, x="bucket", y="count", markers=True, title=f"{label} 댓글량 추이 (KST)")
                st.plotly_chart(fig_ts, use_container_width=True)
            # 좋아요 Top10 (간단)
            best = []
            for chunk in pd.read_csv(csv_path, usecols=["video_id","video_title","author","text","likeCount"], chunksize=200_000):
                chunk["likeCount"] = pd.to_numeric(chunk["likeCount"], errors="coerce").fillna(0).astype(int)
                best.append(chunk.sort_values("likeCount", ascending=False).head(10))
            if best:
                df_top = pd.concat(best).sort_values("likeCount", ascending=False).head(10)
                st.markdown("#### 👍 좋아요 Top10 댓글")
                for _, row in df_top.iterrows():
                    url = f"https://www.youtube.com/watch?v={row['video_id']}"
                    st.markdown(
                        f"<div style='margin-bottom:15px;'>"
                        f"<b>{int(row['likeCount'])} 👍</b> — {row.get('author','')}<br>"
                        f"<span style='font-size:14px;'>▶️ <a href='{url}' target='_blank' style='color:black; text-decoration:none;'>"
                        f"{str(row.get('video_title','(제목없음)'))[:60]}</a></span><br>"
                        f"> {str(row.get('text',''))[:150]}{'…' if len(str(row.get('text','')))>150 else ''}"
                        f"</div>", unsafe_allow_html=True
                    )

            # 다운로드
            st.markdown("---")
            with open(csv_path, "rb") as f:
                st.download_button("⬇️ 전체 댓글 CSV", data=f.read(), file_name=f"chatbot_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

# ===================== 하단 도구 =====================
st.markdown("---")
cols = st.columns(2)
with cols[0]:
    if st.button("🔄 초기화", type="secondary"):
        st.session_state.clear(); safe_rerun()
with cols[1]:
    if st.button("🧹 캐시/메모리 정리"):
        st.cache_data.clear(); gc.collect(); st.success("캐시/메모리 정리 완료")
