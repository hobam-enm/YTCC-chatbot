# -*- coding: utf-8 -*-
# 💬 유튜브 댓글분석기 — 챗봇 모드 (상단 정량 고정 + 하단 대화)
# - 한 줄 요청 → (자연어 해석) → 영상 수집 → 댓글 수집(스트리밍 CSV) → AI 요약
# - 후속질문은 “재수집 없이” 기존 샘플+대화 맥락으로만 답변
# - 상단에 컴팩트 정량+다운로드를 Sticky로 고정, 대화는 아래에서 이어짐

import streamlit as st
import pandas as pd
import os, re, gc, time, json
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai

import plotly.express as px
from plotly import graph_objects as go
import numpy as np
import circlify

# ===================== 0) 페이지 & 전역 =====================
st.set_page_config(page_title="💬 유튜브 댓글분석기: 챗봇", layout="wide", initial_sidebar_state="collapsed")
st.title("💬 유튜브 댓글분석기: 챗봇 모드")

# Sticky 헤더(정량/다운로드)용 CSS
st.markdown(
    """
    <style>
      .sticky-header {
        position: sticky; top: 0; z-index: 999;
        background: white; border-bottom: 1px solid #eee;
        padding: 8px 10px; margin: -1rem -1rem 10px -1rem;
      }
      .mini-kpis .metric { font-size: 12px; color:#6b7280; }
      .mini-kpis .value  { font-size: 18px; font-weight:700; }
      .mini-badge {
        display:inline-block; background:#f3f4f6; border-radius:999px;
        padding:2px 8px; margin: 0 6px 6px 0; font-size:11px; color:#374151;
      }
      .mini-actions .stButton>button, .mini-actions .stDownloadButton>button {
        padding:4px 8px; font-size:12px; height:30px;
      }
    </style>
    """,
    unsafe_allow_html=True
)

BASE_DIR = "/tmp"; os.makedirs(BASE_DIR, exist_ok=True)
KST = timezone(timedelta(hours=9))

def now_kst(): return datetime.now(tz=KST)
def to_iso_kst(dt: datetime) -> str:
    if dt.tzinfo is None: dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST).isoformat(timespec="seconds")
def kst_to_rfc3339_utc(dt_kst: datetime) -> str:
    if dt_kst.tzinfo is None: dt_kst = dt_kst.replace(tzinfo=KST)
    return dt_kst.astimezone(timezone.utc).isoformat().replace("+00:00","Z")

# ===================== 1) 키/상수 =====================
_YT_FALLBACK = []
_GEM_FALLBACK = []
YT_API_KEYS       = list(st.secrets.get("YT_API_KEYS", [])) or _YT_FALLBACK
GEMINI_API_KEYS   = list(st.secrets.get("GEMINI_API_KEYS", [])) or _GEM_FALLBACK
GEMINI_MODEL      = st.secrets.get("GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_TIMEOUT    = int(st.secrets.get("GEMINI_TIMEOUT", 120))
GEMINI_MAX_TOKENS = int(st.secrets.get("GEMINI_MAX_TOKENS", 2048))

MAX_TOTAL_COMMENTS   = 120_000
MAX_COMMENTS_PER_VID = 4_000

# ===================== 2) 상태 =====================
def ensure_state():
    defaults = dict(
        chat=[],                 # [{role, content}]
        last_schema=None,        # dict
        last_csv="",             # collected comments csv path
        last_df=None,            # videos dataframe
        last_total_comments=0,   # 정확값
        last_total_videos=0,     # 정확값
        last_keywords=[],        # 표시용
        last_entities=[],        # 표시용
        last_period=("", ""),    # (start_iso, end_iso)
        sample_text="",          # LLM 샘플(후속질문 재사용)
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
ensure_state()

def safe_rerun():
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(fn): fn()

# ===================== 3) 로테이터/유튜브 래퍼 =====================
class RotatingKeys:
    def __init__(self, keys, state_key: str, on_rotate=None, as_str=True):
        cleaned = []
        for k in (keys or []):
            if k is None: continue
            if as_str and isinstance(k, str):
                ks = k.strip()
                if ks: cleaned.append(ks)
            else:
                cleaned.append(k)
        self.keys = cleaned[:10]
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
    def __init__(self, keys, state_key="yt_key_idx", log=None):
        self.rot = RotatingKeys(keys, state_key, on_rotate=lambda i,k: log and log(f"🔁 YouTube 키 전환 → #{i+1}"))
        self.log = log
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

# ===================== 4) 자연어 해석 프롬프트/파서 =====================
LIGHT_PROMPT = (
    "역할: 유튜브 댓글 반응 분석기의 자연어 해석가.\n"
    "목표: 한국어 입력에서 [기간(KST)]과 [키워드/엔티티/옵션]을 해석.\n"
    "규칙:\n"
    "- 기간은 Asia/Seoul 기준, 상대기간의 종료는 지금.\n"
    "- 작품/인물 고유 표기는 원문 유지.\n"
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

def call_gemini_rotating(model_name, keys, system_instruction, user_payload, timeout_s=120, max_tokens=2048, on_rotate=None) -> str:
    rk = RotatingKeys(keys, "gem_key_idx", on_rotate=lambda i,k: on_rotate and on_rotate(i,k))
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

# ===================== 5) YouTube API =====================
def yt_search_videos(rt, keyword, max_results, order="relevance", published_after=None, published_before=None, log=None):
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
        if log: log(f"검색 진행: {len(video_ids)}개")
        time.sleep(0.3)
    return video_ids

def yt_video_statistics(rt, video_ids, log=None):
    rows = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        if not batch: continue
        resp = rt.execute(lambda s: s.videos().list(part="statistics,snippet,contentDetails", id=",".join(batch)))
        for item in resp.get("items", []):
            stats = item.get("statistics", {}); snip = item.get("snippet", {}); cont = item.get("contentDetails", {})
            dur = cont.get("duration", "")
            def _dsec(d: str):
                if not d or not d.startswith("P"): return None
                h = re.search(r"(\d+)H", d); m = re.search(r"(\d+)M", d); s = re.search(r"(\d+)S", d)
                return (int(h.group(1)) if h else 0)*3600 + (int(m.group(1)) if m else 0)*60 + (int(s.group(1)) if s else 0)
            dur_sec = _dsec(dur)
            short_type = "Shorts" if (dur_sec is not None and dur_sec <= 60) else "Clip"
            vid_id = item.get("id")
            rows.append({
                "video_id": vid_id,
                "video_url": f"https://www.youtube.com/watch?v={vid_id}",
                "title": snip.get("title",""),
                "channelTitle": snip.get("channelTitle",""),
                "publishedAt": snip.get("publishedAt",""),
                "duration": dur,
                "shortType": short_type,
                "viewCount": int(stats.get("viewCount",0) or 0),
                "likeCount": int(stats.get("likeCount",0) or 0),
                "commentCount": int(stats.get("commentCount",0) or 0),
            })
        if log: log(f"통계 배치 {i//50+1} 완료")
        time.sleep(0.3)
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
                prog.progress(min(0.85, frac))
            if total_written >= max_total_comments: break
    return out_csv, total_written

# ===================== 6) LLM 직렬화 =====================
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

# ===================== 7) 정량 시각화(컴팩트) =====================
def _simple_tokens(text):
    return "".join([ch if ("가" <= ch <= "힣") or ch.isalnum() else " " for ch in str(text)]).split()

def keyword_bubble_from_csv(csv_path: str) -> go.Figure | None:
    if not csv_path or not os.path.exists(csv_path): return None
    counts = {}; taken = 0
    for chunk in pd.read_csv(csv_path, usecols=["text"], chunksize=100_000):
        for t in chunk["text"].astype(str).str.slice(0, 200):
            for w in _simple_tokens(t):
                if len(w) < 2: continue
                counts[w] = counts.get(w, 0) + 1
            taken += 1
            if taken >= 100000: break
        if taken >= 100000: break
    if not counts: return None
    df_kw = pd.DataFrame(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:30], columns=["word","count"])
    df_kw["label"] = df_kw["word"] + "<br>" + df_kw["count"].astype(str)
    df_kw["scaled"] = np.sqrt(df_kw["count"])
    circles = circlify.circlify([{"id": w, "datum": s} for w, s in zip(df_kw["word"], df_kw["scaled"])], show_enclosure=False, target_enclosure=circlify.Circle(x=0, y=0, r=1))
    pos = {c.ex["id"]:(c.x,c.y,c.r) for c in circles if "id" in c.ex}
    df_kw["x"] = df_kw["word"].map(lambda w: pos[w][0])
    df_kw["y"] = df_kw["word"].map(lambda w: pos[w][1])
    df_kw["r"] = df_kw["word"].map(lambda w: pos[w][2])
    s_min, s_max = df_kw["scaled"].min(), df_kw["scaled"].max()
    df_kw["font_size"] = df_kw["scaled"].apply(lambda s: int(10 + (s - s_min)/max(s_max - s_min, 1) * 12))
    fig = go.Figure()
    palette = px.colors.sequential.Blues
    df_kw["color_idx"] = df_kw["scaled"].apply(lambda s: int((s - s_min) / max(s_max - s_min, 1) * (len(palette) - 1)))
    for _, row in df_kw.iterrows():
        color = palette[int(row["color_idx"])]
        fig.add_shape(type="circle", xref="x", yref="y", x0=row["x"]-row["r"], y0=row["y"]-row["r"], x1=row["x"]+row["r"], y1=row["y"]+row["r"], line=dict(width=0), fillcolor=color, opacity=0.88, layer="below")
    fig.add_trace(go.Scatter(x=df_kw["x"], y=df_kw["y"], mode="text", text=df_kw["label"], textposition="middle center", textfont=dict(color="white", size=df_kw["font_size"].tolist())))
    fig.update_xaxes(visible=False, range=[-1.05, 1.05])
    fig.update_yaxes(visible=False, range=[-1.05, 1.05], scaleanchor="x", scaleratio=1)
    fig.update_layout(title="Top30 키워드 버블", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=36, b=0), height=280)
    return fig

def timeseries_from_csv(csv_path: str):
    if not csv_path or not os.path.exists(csv_path): return None, None
    tmin = None; tmax = None
    for chunk in pd.read_csv(csv_path, usecols=["publishedAt"], chunksize=200_000):
        dt = pd.to_datetime(chunk["publishedAt"], errors="coerce", utc=True)
        if dt.notna().any():
            lo, hi = dt.min(), dt.max()
            tmin = lo if (tmin is None or (lo < tmin)) else tmin
            tmax = hi if (tmax is None or (hi > tmax)) else tmax
    if tmin is None or tmax is None: return None, None
    span_hours = (tmax - tmin).total_seconds()/3600.0
    use_hour = (span_hours <= 48)
    agg = {}
    for chunk in pd.read_csv(csv_path, usecols=["publishedAt"], chunksize=200_000):
        dt = pd.to_datetime(chunk["publishedAt"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul").dropna()
        if dt.empty: continue
        bucket = (dt.dt.floor("H") if use_hour else dt.dt.floor("D"))
        vc = bucket.value_counts()
        for t, c in vc.items(): agg[t] = agg.get(t, 0) + int(c)
    ts = pd.Series(agg).sort_index().rename("count").reset_index().rename(columns={"index":"bucket"})
    return ts, ("시간별" if use_hour else "일자별")

def render_quant_viz_from_paths(csv_path: str, df_stats: pd.DataFrame, scope_label="(KST)"):
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            fig = keyword_bubble_from_csv(csv_path)
            if fig is not None: st.plotly_chart(fig, use_container_width=True)
            else: st.info("키워드 데이터 없음")
    with col2:
        with st.container(border=True):
            ts, label = timeseries_from_csv(csv_path)
            if ts is not None and not ts.empty:
                fig_ts = px.line(ts, x="bucket", y="count", markers=True, title=f"{label} 댓글량 추이 {scope_label}", height=280)
                st.plotly_chart(fig_ts, use_container_width=True)
            else:
                st.info("시간대별 데이터 없음")

    # 좋아요 Top10 (컴팩트 텍스트 리스트)
    with st.container(border=True):
        st.markdown("**👍 좋아요 Top10 댓글**")
        best = []
        for chunk in pd.read_csv(csv_path, usecols=["video_id","video_title","author","text","likeCount"], chunksize=200_000):
            chunk["likeCount"] = pd.to_numeric(chunk["likeCount"], errors="coerce").fillna(0).astype(int)
            best.append(chunk.sort_values("likeCount", ascending=False).head(10))
        if best:
            df_top = pd.concat(best).sort_values("likeCount", ascending=False).head(10)
            for _, row in df_top.iterrows():
                url = f"https://www.youtube.com/watch?v={row['video_id']}"
                st.markdown(
                    f"<div style='margin-bottom:10px;font-size:13px;'>"
                    f"<b>{int(row['likeCount'])} 👍</b> — {row.get('author','')}<br>"
                    f"<span style='font-size:12px;'>▶️ <a href='{url}' target='_blank' style='color:black; text-decoration:none;'>"
                    f"{str(row.get('video_title','(제목없음)'))[:60]}</a></span><br>"
                    f"> {str(row.get('text',''))[:150]}{'…' if len(str(row.get('text','')))>150 else ''}"
                    f"</div>", unsafe_allow_html=True
                )

    # 영상 Top10 / 작성자 Top10
    if df_stats is not None and not df_stats.empty:
        col3, col4 = st.columns(2)
        with col3:
            with st.container(border=True):
                st.markdown("**🎞️ Top10 영상 댓글수**")
                top_vids = df_stats.sort_values(by="commentCount", ascending=False).head(10).copy()
                if not top_vids.empty:
                    top_vids["title_short"] = top_vids["title"].apply(lambda t: t[:20] + "…" if isinstance(t, str) and len(t) > 20 else t)
                    fig_vids = px.bar(top_vids, x="commentCount", y="title_short", orientation="h", text="commentCount", height=300)
                    st.plotly_chart(fig_vids, use_container_width=True)
        with col4:
            with st.container(border=True):
                st.markdown("**✍️ 댓글 작성자 Top10**")
                counts = {}
                for chunk in pd.read_csv(csv_path, usecols=["author"], chunksize=200_000):
                    vc = chunk["author"].astype(str).value_counts()
                    for k, v in vc.items():
                        counts[k] = counts.get(k, 0) + int(v)
                if counts:
                    ta = pd.Series(counts).sort_values(ascending=False).head(10).reset_index().rename(columns={"index":"author",0:"count"})
                    fig_auth = px.bar(ta, x="count", y="author", orientation="h", text="count", height=300)
                    st.plotly_chart(fig_auth, use_container_width=True)

# ===================== 8) Sticky 정량 헤더 =====================
def render_quant_header_compact():
    if not st.session_state.get("last_csv"):
        return
    total_comments = int(st.session_state.get("last_total_comments", 0))
    total_videos   = int(st.session_state.get("last_total_videos", 0))
    period         = st.session_state.get("last_period", ("?","?"))
    keywords       = st.session_state.get("last_keywords", []) or []
    entities       = st.session_state.get("last_entities", []) or []
    df_stats       = st.session_state.get("last_df")

    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 2.4, 1.4], vertical_alignment="center")
    with c1:
        st.markdown(
            "<div class='mini-kpis'>"
            f"<div class='metric'>총 댓글</div><div class='value'>{total_comments:,}</div>"
            f"<div class='metric' style='margin-top:4px;'>대상 영상</div><div class='value'>{total_videos:,}</div>"
            "</div>", unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<span class='mini-badge'>기간: {period[0]} ~ {period[1]}</span>"
            f"<span class='mini-badge'>키워드: {', '.join(keywords) if keywords else '(없음)'}</span>"
            f"<span class='mini-badge'>엔티티: {', '.join(entities) if entities else '(없음)'}</span>",
            unsafe_allow_html=True
        )
    with c3:
        colx, coly = st.columns(2)
        with colx:
            with open(st.session_state["last_csv"], "rb") as f:
                st.download_button("전체댓글 CSV", f.read(),
                    file_name=f"chatbot_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv", key="dl_comments_top")
        with coly:
            if df_stats is not None and not df_stats.empty:
                csv_videos = df_stats.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button("영상목록 CSV", csv_videos,
                    file_name=f"chatbot_videos_{len(df_stats)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv", key="dl_videos_top")
    with st.expander("📊 정량 하이라이트 (축소)", expanded=False):
        render_quant_viz_from_paths(st.session_state["last_csv"], st.session_state["last_df"], scope_label="(KST)")
    st.markdown('</div>', unsafe_allow_html=True)

# ===================== 9) 대화 렌더 =====================
def render_chat_log():
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# ===================== 10) 실행 파이프라인 (첫 질문) =====================
def run_pipeline_first_turn(user_query: str):
    # 1) 해석
    if not GEMINI_API_KEYS:
        with st.chat_message("assistant"):
            st.markdown("Gemini API Key가 비어 있어요.")
        return
    payload = LIGHT_PROMPT.replace("{USER_QUERY}", user_query)
    with st.status("해석 중…", expanded=False) as status:
        light = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, "", payload, timeout_s=GEMINI_TIMEOUT, max_tokens=GEMINI_MAX_TOKENS, on_rotate=lambda i,k: status.write(f"🔁 Gemini 키 전환 → #{i+1}"))
        schema = parse_light_block_to_schema(light)
        status.update(label="해석 완료", state="complete")

    # 2) 검색/수집
    if not YT_API_KEYS:
        with st.chat_message("assistant"):
            st.markdown("YouTube API Key가 비어 있어요.")
        return
    start_dt = datetime.fromisoformat(schema["start_iso"]).astimezone(KST)
    end_dt   = datetime.fromisoformat(schema["end_iso"]).astimezone(KST)
    published_after = kst_to_rfc3339_utc(start_dt)
    published_before = kst_to_rfc3339_utc(end_dt)
    kw_main  = schema.get("keywords", [])
    kw_ent   = schema.get("entities", [])
    include_replies = bool(schema.get("options",{}).get("include_replies", False))

    with st.status("영상/댓글 수집 중…", expanded=True) as status:
        rt = RotatingYouTube(YT_API_KEYS, log=lambda m: status.write(m))
        all_ids = []
        for base_kw in (kw_main or ["유튜브"]):
            all_ids += yt_search_videos(rt, base_kw, 60, "relevance", published_after, published_before, log=status.write)
            for e in (kw_ent or []):
                all_ids += yt_search_videos(rt, f"{base_kw} {e}", 30, "relevance", published_after, published_before, log=None)
        all_ids = list(dict.fromkeys(all_ids))
        status.write(f"대상 영상: {len(all_ids)}개")

        stats = yt_video_statistics(rt, all_ids, log=status.write)
        df_stats = pd.DataFrame(stats)
        if not df_stats.empty and "publishedAt" in df_stats.columns:
            df_stats["publishedAt_kst"] = (
                pd.to_datetime(df_stats["publishedAt"], errors="coerce", utc=True)
                .dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M:%S")
            )

        prog = st.progress(0.0, text="댓글 수집 중…")
        csv_path, total_cnt = parallel_collect_comments_streaming(
            video_list=df_stats.to_dict('records'),
            rt_keys=YT_API_KEYS,
            include_replies=include_replies,
            max_total_comments=MAX_TOTAL_COMMENTS,
            max_per_video=MAX_COMMENTS_PER_VID,
            prog=prog
        )
        status.write(f"총 댓글 수집: {total_cnt:,}개")
        status.update(label="수집 완료", state="complete")

    if total_cnt == 0:
        with st.chat_message("assistant"):
            st.markdown("지정 기간/키워드에서 댓글이 보이지 않아. 기간/키워드를 조정해줘.")
        return

    # 3) 요약 생성
    a_text, _, _ = serialize_comments_for_llm_from_file(csv_path)
    sys = ("너는 유튜브 댓글을 분석하는 어시스턴트다. "
           "아래 키워드/엔티티와 지정된 기간의 댓글 샘플을 바탕으로 핵심 포인트를 항목화하고, "
           "긍/부/중 비율과 대표 코멘트(10개 미만)를 제시하라.")
    prompt = (
        f"[키워드]: {', '.join(kw_main)}\n"
        f"[엔티티]: {', '.join(kw_ent)}\n"
        f"[기간(KST)]: {schema['start_iso']} ~ {schema['end_iso']}\n\n"
        f"[댓글 샘플]:\n{a_text}\n"
    )
    answer_md = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, sys, prompt, timeout_s=GEMINI_TIMEOUT, max_tokens=GEMINI_MAX_TOKENS)

    # 4) 상태 저장 (정확값 + 컨텍스트)
    st.session_state["last_csv"]            = csv_path
    st.session_state["last_df"]             = df_stats
    st.session_state["last_total_comments"] = int(total_cnt)            # ✅ 정확값
    st.session_state["last_total_videos"]   = int(len(df_stats))        # ✅ 정확값
    st.session_state["last_keywords"]       = kw_main
    st.session_state["last_entities"]       = kw_ent
    st.session_state["last_period"]         = (schema["start_iso"], schema["end_iso"])
    st.session_state["last_schema"]         = schema
    st.session_state["sample_text"]         = a_text

    # 5) 대화 로그 반영(첫 답변)
    with st.chat_message("assistant"):
        st.markdown(
            f"<div style='font-size:12px;color:#6b7280;margin-bottom:6px'>"
            f"분석키워드: {', '.join(kw_main) if kw_main else '(없음)'} · 기간: {schema['start_iso']} ~ {schema['end_iso']}"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(answer_md)
    st.session_state["chat"].append({"role":"assistant","content":answer_md})

# ===================== 11) 후속질문 (재수집 없음) =====================
def run_followup_turn(user_query: str):
    schema = st.session_state.get("last_schema") or {}
    sample_text = st.session_state.get("sample_text","")
    # 최근 대화 맥락 10턴
    lines = []
    for m in st.session_state["chat"][-10:]:
        if m["role"] == "user": lines.append(f"[이전 Q]: {m['content']}")
        else:                   lines.append(f"[이전 A]: {m['content']}")
    context_str = "\n".join(lines)

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
    answer_md = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, sys, payload, timeout_s=GEMINI_TIMEOUT, max_tokens=GEMINI_MAX_TOKENS)

    with st.chat_message("assistant"):
        st.markdown(
            f"<div style='font-size:12px;color:#6b7280;margin-bottom:6px'>"
            f"분석키워드: {', '.join(st.session_state.get('last_keywords',[])) or '(없음)'} · "
            f"기간: {schema.get('start_iso','?')} ~ {schema.get('end_iso','?')}"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(answer_md)
    st.session_state["chat"].append({"role":"assistant","content":answer_md})

# ===================== 12) 상단 고정 정량 헤더 =====================
render_quant_header_compact()

# ===================== 13) 대화 로그 & 입력 =====================
render_chat_log()
prompt = st.chat_input(placeholder="한 줄로 요청하세요 (예: 최근 24시간 폭군의셰프 반응 분석)")

if prompt:
    # 사용자 메시지 로그
    st.session_state["chat"].append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 첫 질문인지/후속인지 판단
    followup = bool(st.session_state.get("last_csv"))
    if followup:
        run_followup_turn(prompt)
    else:
        run_pipeline_first_turn(prompt)

# ===================== 14) 하단 도구 =====================
st.markdown("---")
colx, coly = st.columns(2)
with colx:
    if st.button("🔄 초기화", type="secondary"):
        st.session_state.clear(); safe_rerun()
with coly:
    if st.button("🧹 캐시 정리"):
        st.cache_data.clear(); gc.collect()
        st.success("캐시/메모리 정리 완료")
