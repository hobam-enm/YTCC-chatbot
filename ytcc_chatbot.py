# -*- coding: utf-8 -*-
# 💬 유튜브 댓글분석기 — 챗봇 모드 (대화 상단 + 정량 하단)

import streamlit as st
import pandas as pd
import os, re, gc, time
from datetime import datetime, timedelta, timezone
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

# ===================== 기본 설정 =====================
st.set_page_config(page_title="💬 유튜브 댓글분석기: 챗봇 모드", layout="wide", initial_sidebar_state="collapsed")
st.title("💬 유튜브 댓글분석기: 챗봇 모드")

BASE_DIR = "/tmp"; os.makedirs(BASE_DIR, exist_ok=True)
KST = timezone(timedelta(hours=9))
def now_kst(): return datetime.now(tz=KST)
def to_iso_kst(dt: datetime) -> str:
    if dt.tzinfo is None: dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST).isoformat(timespec="seconds")
def kst_to_rfc3339_utc(dt_kst: datetime) -> str:
    if dt_kst.tzinfo is None: dt_kst = dt_kst.replace(tzinfo=KST)
    return dt_kst.astimezone(timezone.utc).isoformat().replace("+00:00","Z")

MAX_TOTAL_COMMENTS = 120_000
MAX_COMMENTS_PER_VIDEO = 4_000

GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_TIMEOUT = int(st.secrets.get("GEMINI_TIMEOUT", 120))
GEMINI_MAX_TOKENS = int(st.secrets.get("GEMINI_MAX_TOKENS", 2048))

_YT_FALLBACK = []
_GEM_FALLBACK = []
YT_API_KEYS = list(st.secrets.get("YT_API_KEYS", [])) or _YT_FALLBACK
GEMINI_API_KEYS = list(st.secrets.get("GEMINI_API_KEYS", [])) or _GEM_FALLBACK

def safe_rerun():
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(fn): return fn()
    raise RuntimeError("No rerun function available.")

# ===================== 키 로테이터 =====================
class RotatingKeys:
    def __init__(self, keys, state_key: str, on_rotate=None):
        ks = [k.strip() for k in (keys or []) if isinstance(k,str) and k.strip()]
        self.keys = ks[:10]
        self.state_key = state_key
        self.on_rotate = on_rotate
        idx = st.session_state.get(state_key, 0)
        self.idx = 0 if not self.keys else (idx % len(self.keys))
        st.session_state[state_key] = self.idx
    def current(self): return self.keys[self.idx] if self.keys else None
    def rotate(self):
        if not self.keys: return
        self.idx = (self.idx + 1) % len(self.keys)
        st.session_state[self.state_key] = self.idx
        if callable(self.on_rotate): self.on_rotate(self.idx, self.current())

class RotatingYouTube:
    def __init__(self, keys, state_key="yt_key_idx"):
        self.rot = RotatingKeys(keys, state_key)
        key = self.rot.current()
        if not key: raise RuntimeError("YouTube API Key가 비어 있습니다.")
        self.svc = build("youtube","v3",developerKey=key)
    def execute(self, factory):
        try:
            return factory(self.svc).execute()
        except HttpError as e:
            status = getattr(getattr(e,'resp',None),'status',None)
            msg = (getattr(e,'content',b'').decode('utf-8','ignore') or '').lower()
            if status in (403,429) and any(t in msg for t in ["quota","rate","limit"]) and len(YT_API_KEYS)>1:
                self.rot.rotate(); self.svc = build("youtube","v3",developerKey=self.rot.current())
                return factory(self.svc).execute()
            raise

def is_gemini_quota_error(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return ("429" in msg) or ("too many requests" in msg) or ("rate limit" in msg) or ("resource exhausted" in msg) or ("quota" in msg)

def call_gemini_rotating(model_name, keys, system_instruction, user_payload,
                         timeout_s=GEMINI_TIMEOUT, max_tokens=GEMINI_MAX_TOKENS, on_rotate=None) -> str:
    rot = RotatingKeys(keys, state_key="gem_key_idx", on_rotate=lambda i,k: on_rotate and on_rotate(i,k))
    if not rot.current(): raise RuntimeError("Gemini API Key가 비어 있습니다.")
    attempts = 0; max_attempts = len(rot.keys) if rot.keys else 1
    while attempts < max_attempts:
        try:
            genai.configure(api_key=rot.current())
            model = genai.GenerativeModel(model_name, generation_config={"temperature":0.2,"max_output_tokens":max_tokens,"top_p":0.9})
            resp = model.generate_content([system_instruction, user_payload], request_options={"timeout": timeout_s})
            out = getattr(resp,"text",None)
            if not out and getattr(resp,"candidates",None):
                parts = getattr(resp.candidates[0].content,"parts",None)
                if parts and hasattr(parts[0],"text"): out = parts[0].text
            return out or ""
        except Exception as e:
            if is_gemini_quota_error(e) and len(rot.keys)>1:
                rot.rotate(); attempts += 1; continue
            raise

# ===================== 해석 프롬프트/파서 =====================
LIGHT_PROMPT = (
    "역할: ‘유튜브 댓글 반응 분석기’의 자연어 해석가.\n"
    "목표: 요청에서 [기간(KST)]과 [키워드/엔티티] 및 옵션을 해석.\n"
    "규칙: 상대기간의 종료=지금(KST), 절대기간은 그대로.\n"
    "출력 6줄 고정:\n"
    "- 한 줄 요약: <문장>\n"
    "- 기간(KST): <YYYY-MM-DDTHH:MM:SS+09:00> ~ <YYYY-MM-DDTHH:MM:SS+09:00>\n"
    "- 키워드: [<메인1>, <메인2>…]\n"
    "- 엔티티/보조: [<보조들>]\n"
    "- 옵션: { include_replies: true|false, channel_filter: \"any|official|unofficial\", lang: \"ko|en|auto\" }\n"
    "- 원문: {USER_QUERY}\n\n"
    f"현재 KST: {to_iso_kst(now_kst())}\n입력:\n{{USER_QUERY}}"
)

def parse_light_block_to_schema(light_text: str) -> dict:
    raw = (light_text or "").strip()
    m_time = re.search(r"기간\(KST\)\s*:\s*([^~]+)~\s*([^\n]+)", raw)
    start_iso = m_time.group(1).strip() if m_time else None
    end_iso   = m_time.group(2).strip() if m_time else None
    m_kw = re.search(r"키워드\s*:\s*\[(.*?)\]", raw, flags=re.DOTALL)
    keywords = []
    if m_kw:
        for part in re.split(r"\s*,\s*", m_kw.group(1)):
            part = re.sub(r"\(.*?\)", "", part).strip()
            if part: keywords.append(part)
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

# ===================== YouTube 검색/통계/댓글 =====================
def yt_search_videos(rt, keyword, max_results, order="relevance", published_after=None, published_before=None):
    vids, token = [], None
    while len(vids) < max_results:
        params = dict(q=keyword, part="id", type="video", order=order, maxResults=min(50, max_results-len(vids)))
        if published_after: params["publishedAfter"] = published_after
        if published_before: params["publishedBefore"] = published_before
        if token: params["pageToken"] = token
        resp = rt.execute(lambda s: s.search().list(**params))
        for it in resp.get("items", []):
            vid = it["id"]["videoId"]
            if vid not in vids: vids.append(vid)
        token = resp.get("nextPageToken")
        if not token: break
        time.sleep(0.25)
    return vids

def yt_video_statistics(rt, video_ids):
    rows = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        if not batch: continue
        # 🔧 SyntaxError fix: id=",".join(batch)
        resp = rt.execute(lambda s: s.videos().list(part="statistics,snippet,contentDetails", id=",".join(batch)))
        for item in resp.get("items", []):
            stats = item.get("statistics", {}); snip = item.get("snippet", {}); cont = item.get("contentDetails", {})
            dur_iso = cont.get("duration", "")
            def _dsec(dur:str):
                if not dur or not dur.startswith("P"): return None
                h = re.search(r"(\d+)H", dur); m = re.search(r"(\d+)M", dur); s = re.search(r"(\d+)S", dur)
                return (int(h.group(1)) if h else 0)*3600 + (int(m.group(1)) if m else 0)*60 + (int(s.group(1)) if s else 0)
            dur_sec = _dsec(dur_iso)
            short_type = "Shorts" if (dur_sec is not None and dur_sec<=60) else "Clip"
            vid_id = item.get("id")
            rows.append({
                "video_id": vid_id,
                "video_url": f"https://www.youtube.com/watch?v={vid_id}",
                "title": snip.get("title",""),
                "channelTitle": snip.get("channelTitle",""),
                "publishedAt": snip.get("publishedAt",""),
                "duration": dur_iso,
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
                frac = 0.35 + (done/total_videos) * 0.50  # 35%→85%
                prog.progress(min(0.85, frac), text="댓글 수집중…")
            if total_written >= max_total_comments: break
    return out_csv, total_written

# ===================== LLM 직렬화 =====================
def serialize_comments_for_llm_from_file(csv_path: str, max_rows=1500, max_chars_per_comment=280, max_total_chars=420_000):
    if not csv_path or not os.path.exists(csv_path): return "", 0, 0
    lines, total = [], 0; remaining = max_rows
    for chunk in pd.read_csv(csv_path, chunksize=120_000):
        if "likeCount" in chunk.columns:
            chunk = chunk.sort_values("likeCount", ascending=False)
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

# ===================== 정량 (간단 지표) =====================
def timeseries_from_file(csv_path: str):
    if not csv_path or not os.path.exists(csv_path): return None, None
    tmin = None; tmax = None
    for chunk in pd.read_csv(csv_path, usecols=["publishedAt"], chunksize=200_000):
        dt = pd.to_datetime(chunk["publishedAt"], errors="coerce", utc=True)
        if dt.notna().any():
            tmin = dt.min() if (tmin is None or dt.min()<tmin) else tmin
            tmax = dt.max() if (tmax is None or dt.max()>tmax) else tmax
    if tmin is None or tmax is None: return None, None
    span_hours = (tmax - tmin).total_seconds()/3600.0
    use_hour = (span_hours <= 48)
    agg = {}
    for chunk in pd.read_csv(csv_path, usecols=["publishedAt"], chunksize=200_000):
        dt = pd.to_datetime(chunk["publishedAt"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")
        dt = dt.dropna()
        if dt.empty: continue
        bucket = (dt.dt.floor("H") if use_hour else dt.dt.floor("D"))
        vc = bucket.value_counts()
        for t, c in vc.items(): agg[t] = agg.get(t, 0) + int(c)
    ts = pd.Series(agg).sort_index().rename("count").reset_index().rename(columns={"index":"bucket"})
    return ts, ("시간별" if use_hour else "일자별")

# ===================== 상태 =====================
def ensure_state():
    defaults = dict(
        chat=[],                 # [(role, text_md, meta_html)]
        schema=None,             # 마지막 해석 스키마
        sample_text="",          # 직렬화된 댓글 샘플(후속질문 재사용)
        last_csv="",             # 수집 CSV 경로
        last_df=None,            # 영상 메타 DF
    )
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v
ensure_state()

def render_chat():
    for role, text, meta in st.session_state['chat']:
        cls = 'user' if role=='user' else 'assistant'
        st.markdown(
            f"<div style='max-width:900px;margin:0 auto;'>"
            f"<div style='display:flex;justify-content:{'flex-end' if cls=='user' else 'flex-start'};margin:6px 0;'>"
            f"<div style='max-width:85%;background:{'#e5f0ff' if cls=='user' else '#f8fafc'};padding:10px 12px;border-radius:12px;font-size:14px;line-height:1.5;'>"
            f"{meta or ''}{(text or '').replace('\n','<br>')}"
            f"</div></div></div>", unsafe_allow_html=True
        )

# ===================== 상단: 챗 입력 + 실행 =====================
render_chat()
with st.container(border=True):
    user_query = st.text_input(" ", placeholder="예) 최근 12시간 태풍상사 김준호 댓글반응 분석해줘", label_visibility="collapsed", key="cb_query")
    run = st.button("▶ 실행", type="primary", use_container_width=True)

# ===================== 실행 로직 =====================
if run and user_query:
    # 1) 대화 로그에 사용자 입력 추가
    st.session_state['chat'].append(('user', user_query, None))
    prog = st.progress(0.0, text="해석중…")

    # ===== 첫 질문(수집 필요) vs 후속 질문(수집 불필요) 판단 =====
    is_followup = bool(st.session_state.get('last_csv'))

    if not is_followup:
        # ---- 해석 ----
        if not GEMINI_API_KEYS:
            st.session_state['chat'].append(('assistant', 'Gemini API Key가 없습니다.', None)); st.stop()
        light_text = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, "", LIGHT_PROMPT.replace("{USER_QUERY}", user_query))
        schema = parse_light_block_to_schema(light_text)
        st.session_state['schema'] = schema
        prog.progress(0.20, text="영상 검색중…")

        # ---- 검색/통계 ----
        if not YT_API_KEYS:
            st.session_state['chat'].append(('assistant', 'YouTube API Key가 없습니다.', None)); st.stop()
        start_dt = datetime.fromisoformat(schema["start_iso"]).astimezone(KST)
        end_dt   = datetime.fromisoformat(schema["end_iso"]).astimezone(KST)
        published_after = kst_to_rfc3339_utc(start_dt)
        published_before = kst_to_rfc3339_utc(end_dt)
        keywords = schema.get("keywords", []); entities = schema.get("entities", [])
        include_replies = bool(schema.get("options",{}).get("include_replies", False))

        rt = RotatingYouTube(YT_API_KEYS)
        all_ids = []
        for base_kw in (keywords or ["유튜브"]):
            all_ids += yt_search_videos(rt, base_kw, 60, "relevance", published_after, published_before)
            for e in (entities or []):
                all_ids += yt_search_videos(rt, f"{base_kw} {e}", 30, "relevance", published_after, published_before)
        all_ids = list(dict.fromkeys(all_ids))
        df_stats = pd.DataFrame(yt_video_statistics(rt, all_ids))
        st.session_state['last_df'] = df_stats

        # ---- 댓글 수집 ----
        prog.progress(0.35, text="댓글 수집중…")
        csv_path, total_cnt = parallel_collect_comments_streaming(
            video_list=df_stats.to_dict('records') if not df_stats.empty else [],
            rt_keys=YT_API_KEYS,
            include_replies=include_replies,
            max_total_comments=MAX_TOTAL_COMMENTS,
            max_per_video=MAX_COMMENTS_PER_VIDEO,
            prog=prog
        )
        st.session_state['last_csv'] = csv_path
        if total_cnt == 0:
            prog.progress(1.0, text="완료")
            st.session_state['chat'].append(('assistant', '수집된 댓글이 없습니다. 기간/키워드를 조정해 주세요.', None))
            st.stop()

        # ---- 댓글 샘플 직렬화 ----
        prog.progress(0.90, text="AI 분석중…")
        sample_text, _, _ = serialize_comments_for_llm_from_file(csv_path)
        st.session_state['sample_text'] = sample_text

        # ---- 첫 AI 요약 ----
        sys = ("너는 유튜브 댓글을 분석하는 어시스턴트다. "
               "아래 키워드/엔티티와 지정된 기간의 댓글 샘플을 바탕으로 핵심 포인트를 항목화하고, "
               "긍/부/중 비율과 대표 코멘트(10개 미만)를 제시하라. 반드시 샘플을 근거로 작성.")
        payload = (
            f"[키워드]: {', '.join(keywords)}\n"
            f"[엔티티]: {', '.join(entities)}\n"
            f"[기간(KST)]: {schema['start_iso']} ~ {schema['end_iso']}\n\n"
            f"[댓글 샘플]:\n{sample_text}\n"
        )
        answer_md = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, sys, payload)
        if not answer_md.strip():
            answer_md = "_AI 요약 응답이 비었습니다. 질문을 조금 더 구체적으로 입력해 보세요._"

        meta_html = (
            f"<div style='font-size:12px;color:#6b7280;margin-bottom:6px;'>"
            f"<span style='background:#f3f4f6;border-radius:999px;padding:2px 8px;margin-right:6px;'>분석키워드: {', '.join(keywords) if keywords else '(없음)'}</span>"
            f"<span style='background:#f3f4f6;border-radius:999px;padding:2px 8px;'>기간: {schema['start_iso']} ~ {schema['end_iso']}</span>"
            f"</div>"
        )
        st.session_state['chat'].append(('assistant', answer_md, meta_html))
        prog.progress(1.0, text="완료")

    else:
        # ===== 후속질문: 재수집 없이, 고정 샘플+대화맥락 =====
        schema = st.session_state.get('schema') or {}
        sample_text = st.session_state.get('sample_text', '')
        # 최근 대화 일부를 맥락으로 전달
        ctx_lines = []
        for role, text, _ in st.session_state['chat'][-10:]:
            if role == 'user': ctx_lines.append(f"[이전 Q]: {text}")
            else:             ctx_lines.append(f"[이전 A]: {text}")
        context_str = "\n".join(ctx_lines)

        sys = ("너는 유튜브 댓글을 분석하는 어시스턴트다. "
               "아래는 직렬화된 댓글 샘플(고정)과 이전 대화 맥락이다. "
               "현재 질문에 대해 간결하고 구조화된 답을 한국어로 하라. "
               "반드시 댓글 샘플을 근거로 답하고, 인용 예시는 5개 이하로 제시하라.")
        payload = (
            context_str + "\n\n" +
            f"[현재 질문]: {user_query}\n"
            f"[기간(KST)]: {schema.get('start_iso','?')} ~ {schema.get('end_iso','?')}\n\n"
            f"[댓글 샘플]:\n{sample_text}\n"
        )
        answer_md = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, sys, payload)
        if not answer_md.strip():
            answer_md = "_응답이 비었습니다. 질문을 조금 더 구체적으로 입력해 보세요._"
        meta_html = (
            f"<div style='font-size:12px;color:#6b7280;margin-bottom:6px;'>"
            f"<span style='background:#f3f4f6;border-radius:999px;padding:2px 8px;margin-right:6px;'>분석키워드: {', '.join(schema.get('keywords', [])) or '(없음)'}</span>"
            f"<span style='background:#f3f4f6;border-radius:999px;padding:2px 8px;'>기간: {schema.get('start_iso','?')} ~ {schema.get('end_iso','?')}</span>"
            f"</div>"
        )
        st.session_state['chat'].append(('assistant', answer_md, meta_html))
        prog.progress(1.0, text="완료")

# ===================== 하단: 정량 & 다운로드 (항상 유지) =====================
if st.session_state.get("last_csv"):
    st.markdown("---")
    st.subheader("📊 정량 하이라이트")

    # ① 키워드 버블(간단 토크나이저) — 빠른 표시
    try:
        def _simple_tokens(text):
            return "".join([ch if ("가" <= ch <= "힣") or ch.isalnum() else " " for ch in text]).split()
        counts = {}; taken = 0
        for chunk in pd.read_csv(st.session_state["last_csv"], usecols=["text"], chunksize=100_000):
            for t in chunk["text"].astype(str).str.slice(0, 200):
                for w in _simple_tokens(t):
                    if len(w) < 2: continue
                    counts[w] = counts.get(w, 0) + 1
                taken += 1
                if taken >= 100000: break
            if taken >= 100000: break
        if counts:
            df_kw = pd.DataFrame(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:30], columns=["word","count"])
            df_kw["label"] = df_kw["word"] + "<br>" + df_kw["count"].astype(str)
            df_kw["scaled"] = np.sqrt(df_kw["count"])
            circles = circlify.circlify([{"id": w, "datum": s} for w, s in zip(df_kw["word"], df_kw["scaled"])],
                                        show_enclosure=False, target_enclosure=circlify.Circle(x=0, y=0, r=1))
            pos = {c.ex["id"]:(c.x,c.y,c.r) for c in circles if "id" in c.ex}
            df_kw["x"] = df_kw["word"].map(lambda w: pos[w][0])
            df_kw["y"] = df_kw["word"].map(lambda w: pos[w][1])
            df_kw["r"] = df_kw["word"].map(lambda w: pos[w][2])
            s_min, s_max = df_kw["scaled"].min(), df_kw["scaled"].max()
            df_kw["font_size"] = df_kw["scaled"].apply(lambda s: int(10 + (s - s_min)/max(s_max - s_min, 1) * 12))
            fig_kw = go.Figure()
            palette = px.colors.sequential.Blues
            df_kw["color_idx"] = df_kw["scaled"].apply(lambda s: int((s - s_min)/max(s_max - s_min, 1) * (len(palette) - 1)))
            for _, row in df_kw.iterrows():
                color = palette[int(row["color_idx"])]
                fig_kw.add_shape(type="circle", xref="x", yref="y",
                                 x0=row["x"]-row["r"], y0=row["y"]-row["r"],
                                 x1=row["x"]+row["r"], y1=row["y"]+row["r"],
                                 line=dict(width=0), fillcolor=color, opacity=0.88, layer="below")
            fig_kw.add_trace(go.Scatter(x=df_kw["x"], y=df_kw["y"], mode="text",
                              text=df_kw["label"], textposition="middle center",
                              textfont=dict(color="white", size=df_kw["font_size"].tolist())))
            fig_kw.update_xaxes(visible=False, range=[-1.05, 1.05])
            fig_kw.update_yaxes(visible=False, range=[-1.05, 1.05], scaleanchor="x", scaleratio=1)
            fig_kw.update_layout(title="Top30 키워드 버블", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_kw, use_container_width=True)
    except Exception as e:
        st.info(f"키워드 버블 생성 실패: {e}")

    # ② 시점별 댓글량 추이
    ts, label = timeseries_from_file(st.session_state["last_csv"])
    if ts is not None and not ts.empty:
        fig_ts = px.line(ts, x="bucket", y="count", markers=True, title=f"{label} 댓글량 추이 (KST)")
        st.plotly_chart(fig_ts, use_container_width=True)

    # ③ 좋아요 Top10 댓글
    best = []
    for chunk in pd.read_csv(st.session_state["last_csv"], usecols=["video_id","video_title","author","text","likeCount"], chunksize=200_000):
        chunk["likeCount"] = pd.to_numeric(chunk["likeCount"], errors="coerce").fillna(0).astype(int)
        best.append(chunk.sort_values("likeCount", ascending=False).head(10))
    if best:
        df_top = pd.concat(best).sort_values("likeCount", ascending=False).head(10)
        st.markdown("#### 👍 좋아요 Top10 댓글")
        for _, row in df_top.iterrows():
            url = f"https://www.youtube.com/watch?v={row['video_id']}"
            st.markdown(
                f"<div style='margin-bottom:12px;'>"
                f"<b>{int(row['likeCount'])} 👍</b> — {row.get('author','')}<br>"
                f"<span style='font-size:14px;'>▶️ <a href='{url}' target='_blank' style='color:black; text-decoration:none;'>"
                f"{str(row.get('video_title','(제목없음)'))[:60]}</a></span><br>"
                f"> {str(row.get('text',''))[:150]}{'…' if len(str(row.get('text','')))>150 else ''}"
                f"</div>", unsafe_allow_html=True
            )

    # ④ Top10 영상 댓글수 / ⑤ 작성자 활동량 Top10
    df_stats = st.session_state.get("last_df")
    if df_stats is not None and not df_stats.empty:
        colx, coly = st.columns(2)
        with colx:
            top_vids = df_stats.sort_values(by="commentCount", ascending=False).head(10).copy()
            if not top_vids.empty:
                top_vids["title_short"] = top_vids["title"].apply(lambda t: t[:20] + "…" if isinstance(t, str) and len(t) > 20 else t)
                fig_vids = px.bar(top_vids, x="commentCount", y="title_short", orientation="h", text="commentCount", title="Top10 영상 댓글수")
                st.plotly_chart(fig_vids, use_container_width=True)
        with coly:
            counts = {}
            for chunk in pd.read_csv(st.session_state["last_csv"], usecols=["author"], chunksize=200_000):
                vc = chunk["author"].astype(str).value_counts()
                for k, v in vc.items(): counts[k] = counts.get(k, 0) + int(v)
            if counts:
                ta = pd.Series(counts).sort_values(ascending=False).head(10).reset_index().rename(columns={"index":"author",0:"count"})
                fig_auth = px.bar(ta, x="count", y="author", orientation="h", text="count", title="Top10 댓글 작성자 활동량")
                st.plotly_chart(fig_auth, use_container_width=True)

    # 다운로드
    st.markdown("---")
    with open(st.session_state["last_csv"], "rb") as f:
        st.download_button("⬇️ 전체 댓글 CSV", data=f.read(),
                           file_name=f"chatbot_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")
    if df_stats is not None and not df_stats.empty:
        csv_videos = df_stats.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("⬇️ 전체 영상목록 CSV", data=csv_videos,
                           file_name=f"chatbot_videos_{len(df_stats)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")

# ===================== 하단 도구 =====================
st.markdown("---")
cols = st.columns(2)
with cols[0]:
    if st.button("🔄 초기화", type="secondary"):
        st.session_state.clear(); safe_rerun()
with cols[1]:
    if st.button("🧹 캐시/메모리 정리"):
        st.cache_data.clear(); gc.collect(); st.success("캐시/메모리 정리 완료")
