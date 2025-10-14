# -*- coding: utf-8 -*-
# 💬 유튜브 댓글 분석 챗봇 — 완성본
# - 자연어 한 줄 → (기간/키워드/옵션) 해석(제미나이) → 영상 수집(YouTube API) → 댓글 수집(CSV 스트리밍) → AI 요약 + 정량 시각화
# - 첫 질문: 전체 파이프라인 실행 / 후속 질문: 재수집 없이 직렬화 샘플 + 대화맥락 기반 답변만

import streamlit as st
import pandas as pd
import os, re, gc, time, json
from urllib.parse import urlparse, parse_qs
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
from kiwipiepy import Kiwi
import stopwordsiso as stopwords

# ============== 0) 페이지/기본 ==============
st.set_page_config(page_title="💬 유튜브 댓글 분석 챗봇", layout="wide")
st.title("💬 유튜브 댓글 분석 챗봇")

BASE_DIR = "/tmp"; os.makedirs(BASE_DIR, exist_ok=True)
KST = timezone(timedelta(hours=9))
def now_kst(): return datetime.now(tz=KST)
def to_iso_kst(dt: datetime) -> str:
    if dt.tzinfo is None: dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST).isoformat(timespec="seconds")
def kst_to_rfc3339_utc(dt_kst: datetime) -> str:
    if dt_kst.tzinfo is None: dt_kst = dt_kst.replace(tzinfo=KST)
    return dt_kst.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def safe_rerun():
    try:
        st.rerun()
    except Exception:
        pass

# Limits
MAX_TOTAL_COMMENTS = 120_000
MAX_COMMENTS_PER_VIDEO = 4_000

# ============== 1) 키/모델 세팅 ==============
# YouTube API keys (list)
_YT_FALLBACK = []
YT_API_KEYS = list(st.secrets.get("YT_API_KEYS", [])) or _YT_FALLBACK

# Gemini keys (list or single)
_GEM_FALLBACK = []
GEMINI_API_KEYS = list(st.secrets.get("GEMINI_API_KEYS", [])) or _GEM_FALLBACK
if not GEMINI_API_KEYS:
    single = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY","")).strip()
    if single:
        GEMINI_API_KEYS = [single]

GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_TIMEOUT = int(st.secrets.get("GEMINI_TIMEOUT", 120))
GEMINI_MAX_TOKENS = int(st.secrets.get("GEMINI_MAX_TOKENS", 2048))

# ============== 2) 로테이터/YouTube 래퍼 ==============
class RotatingKeys:
    def __init__(self, keys, state_key: str):
        self.keys = [k.strip() for k in (keys or []) if isinstance(k, str) and k.strip()][:10]
        self.state_key = state_key
        idx = st.session_state.get(state_key, 0)
        self.idx = 0 if not self.keys else (idx % len(self.keys))
        st.session_state[state_key] = self.idx
    def current(self):
        return self.keys[self.idx] if self.keys else None
    def rotate(self):
        if not self.keys: return
        self.idx = (self.idx + 1) % len(self.keys)
        st.session_state[self.state_key] = self.idx

class RotatingYouTube:
    def __init__(self, keys):
        self.rot = RotatingKeys(keys, "yt_key_idx")
        key = self.rot.current()
        if not key:
            raise RuntimeError("YouTube API Key가 비어 있습니다.")
        self.svc = build("youtube", "v3", developerKey=key)
    def execute(self, factory):
        try:
            return factory(self.svc).execute()
        except HttpError as e:
            status = getattr(getattr(e, 'resp', None), 'status', None)
            msg = (getattr(e, 'content', b'').decode('utf-8', 'ignore') or '').lower()
            if status in (403, 429) and any(w in msg for w in ["quota","limit","rate"]) and len(YT_API_KEYS)>1:
                self.rot.rotate(); self.svc = build("youtube","v3",developerKey=self.rot.current())
                return factory(self.svc).execute()
            raise

# ============== 3) Gemini 호출 ==============
def call_gemini_text(system_instruction: str, user_payload: str) -> str:
    rk = RotatingKeys(GEMINI_API_KEYS, "gem_key_idx")
    if not rk.current():
        raise RuntimeError("Gemini API Key가 비어 있습니다.")
    genai.configure(api_key=rk.current())
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={"temperature":0.2, "max_output_tokens":GEMINI_MAX_TOKENS, "top_p":0.9}
    )
    try:
        resp = model.generate_content([system_instruction, user_payload], request_options={"timeout": GEMINI_TIMEOUT})
        return getattr(resp, "text", "") or ""
    except Exception:
        if len(GEMINI_API_KEYS) > 1:
            rk.rotate(); genai.configure(api_key=rk.current())
            model = genai.GenerativeModel(
                GEMINI_MODEL,
                generation_config={"temperature":0.2, "max_output_tokens":GEMINI_MAX_TOKENS, "top_p":0.9}
            )
            resp = model.generate_content([system_instruction, user_payload], request_options={"timeout": GEMINI_TIMEOUT})
            return getattr(resp, "text", "") or ""
        raise

# 자연어 → 라이트 요약 블록
LIGHT_PROMPT = (
    "역할: ‘유튜브 댓글 분석기’의 자연어 해석가.\n"
    "목표: 한국어 요청에서 [검색 기간(KST)]과 [검색 키워드(메인/보조)] 및 옵션을 해석.\n"
    "규칙:\n- 기간은 Asia/Seoul(+09:00) 기준. 상대기간의 종료는 지금.\n- 의미 있는 고유명사 원문 보존.\n"
    "출력(6줄 고정):\n"
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

# ============== 4) YouTube 검색/통계/댓글 ==============
def yt_search_videos(rt, keyword, max_results, order="relevance", published_after=None, published_before=None, log=None):
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
        if log: log(f"검색 누계: {len(vids)}")
        time.sleep(0.25)
    return vids

def yt_video_statistics(rt, video_ids, log=None):
    rows = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        if not batch: continue
        resp = rt.execute(lambda s: s.videos().list(part="statistics,snippet,contentDetails", id=",".join(batch)))
        for item in resp.get("items", []):
            stats = item.get("statistics", {}); snip = item.get("snippet", {}); cont = item.get("contentDetails", {})
            dur_iso = cont.get("duration", "")
            def _dsec(d: str):
                if not d or not d.startswith("P"): return None
                h = re.search(r"(\d+)H", d); m = re.search(r"(\d+)M", d); s = re.search(r"(\d+)S", d)
                return (int(h.group(1)) if h else 0)*3600 + (int(m.group(1)) if m else 0)*60 + (int(s.group(1)) if s else 0)
            dur_sec = _dsec(dur_iso)
            short_type = "Shorts" if (dur_sec is not None and dur_sec<=60) else "Clip"
            vid_id = item.get("id")
            rows.append({
                "video_id": vid_id,
                "video_url": f"https://www.youtube.com/watch?v={vid_id}",
                "title": snip.get("title", ""),
                "channelTitle": snip.get("channelTitle",""),
                "publishedAt": snip.get("publishedAt",""),
                "duration": dur_iso,
                "shortType": short_type,
                "viewCount": int(stats.get("viewCount",0) or 0),
                "likeCount": int(stats.get("likeCount",0) or 0),
                "commentCount": int(stats.get("commentCount",0) or 0),
            })
        if log: log(f"통계 배치 {i//50 + 1} 완료")
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
            if prog:
                frac = 0.35 + (done/total_videos) * 0.50  # 10% 해석, 35% 검색, 85% 수집
                prog.progress(min(0.85, frac))
            if total_written >= max_total_comments: break
    return out_csv, total_written

# ============== 5) LLM 샘플 직렬화 ==============
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

# ============== 6) 정량 시각화 ==============
kiwi = Kiwi()
korean_stopwords = stopwords.stopwords("ko")

@st.cache_data(ttl=600, show_spinner=False)
def compute_keyword_counter_from_file(csv_path: str, stopset_list: list[str], per_comment_cap: int = 200) -> list[tuple[str,int]]:
    if not csv_path or not os.path.exists(csv_path): return []
    stopset = set(stopset_list)
    counter = Counter()
    for chunk in pd.read_csv(csv_path, usecols=["text"], chunksize=100_000):
        texts = (chunk["text"].astype(str).str.slice(0, per_comment_cap)).tolist()
        if not texts: continue
        tokens = kiwi.tokenize(" ".join(texts), normalize_coda=True)
        words = [t.form for t in tokens if t.tag in ("NNG","NNP") and len(t.form) > 1 and t.form not in stopset]
        counter.update(words)
    return counter.most_common(300)

def keyword_bubble_figure_from_counter(counter_items: list[tuple[str,int]]) -> go.Figure | None:
    if not counter_items: return None
    df_kw = pd.DataFrame(counter_items[:30], columns=["word","count"])
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
    fig = go.Figure()
    palette = px.colors.sequential.Blues
    df_kw["color_idx"] = df_kw["scaled"].apply(lambda s: int((s - s_min)/max(s_max - s_min, 1) * (len(palette) - 1)))
    for _, row in df_kw.iterrows():
        color = palette[int(row["color_idx"])]
        fig.add_shape(type="circle", xref="x", yref="y",
                      x0=row["x"]-row["r"], y0=row["y"]-row["r"],
                      x1=row["x"]+row["r"], y1=row["y"]+row["r"],
                      line=dict(width=0), fillcolor=color, opacity=0.88, layer="below")
    fig.add_trace(go.Scatter(x=df_kw["x"], y=df_kw["y"], mode="text",
                             text=df_kw["label"], textposition="middle center",
                             textfont=dict(color="white", size=df_kw["font_size"].tolist())))
    fig.update_xaxes(visible=False, range=[-1.05, 1.05])
    fig.update_yaxes(visible=False, range=[-1.05, 1.05], scaleanchor="x", scaleratio=1)
    fig.update_layout(title="Top30 키워드 버블", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=36, b=0))
    return fig

def timeseries_from_file(csv_path: str):
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
        dt = pd.to_datetime(chunk["publishedAt"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")
        dt = dt.dropna()
        if dt.empty: continue
        bucket = (dt.dt.floor("H") if use_hour else dt.dt.floor("D"))
        vc = bucket.value_counts()
        for t, c in vc.items(): agg[t] = agg.get(t, 0) + int(c)
    ts = pd.Series(agg).sort_index().rename("count").reset_index().rename(columns={"index":"bucket"})
    return ts, ("시간별" if use_hour else "일자별")

def top_authors_from_file(csv_path: str, topn=10):
    if not csv_path or not os.path.exists(csv_path): return None
    counts = {}
    for chunk in pd.read_csv(csv_path, usecols=["author"], chunksize=200_000):
        vc = chunk["author"].astype(str).value_counts()
        for k, v in vc.items(): counts[k] = counts.get(k, 0) + int(v)
    if not counts: return None
    s = pd.Series(counts).sort_values(ascending=False).head(topn)
    return s.reset_index().rename(columns={"index": "author", 0: "count"}).rename(columns={"count": "count"})

def render_quant_viz_from_paths(comments_csv_path: str, df_stats: pd.DataFrame, query_kw: str = ""):
    if not comments_csv_path or not os.path.exists(comments_csv_path): return
    st.markdown("### 📊 정량 하이라이트")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**① 키워드 버블**")
            try:
                custom_stop = {"아","휴","영상","댓글","오늘","진짜","정말","이번","유튜브","부분"}
                stopset = set(korean_stopwords); stopset.update(custom_stop)
                if query_kw.strip():
                    tokens_q = kiwi.tokenize(query_kw, normalize_coda=True)
                    query_words = [t.form for t in tokens_q if t.tag in ("NNG","NNP") and len(t.form) > 1]
                    stopset.update(query_words)
                items = compute_keyword_counter_from_file(comments_csv_path, list(stopset), per_comment_cap=200)
                fig = keyword_bubble_figure_from_counter(items)
                if fig is None:
                    st.info("표시할 키워드가 없습니다.")
                else:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"키워드 분석 불가: {e}")
    with col2:
        with st.container(border=True):
            st.markdown("**② 시점별 댓글량 추이**")
            ts, label = timeseries_from_file(comments_csv_path)
            if ts is not None:
                fig_ts = px.line(ts, x="bucket", y="count", markers=True, title=f"{label} 댓글량 (KST)")
                st.plotly_chart(fig_ts, use_container_width=True)
            else:
                st.info("데이터 없음")

    if df_stats is not None and not df_stats.empty:
        col3, col4 = st.columns(2)
        with col3:
            with st.container(border=True):
                st.markdown("**③ Top10 영상 댓글수**")
                top_vids = df_stats.sort_values(by="commentCount", ascending=False).head(10).copy()
                if not top_vids.empty:
                    top_vids["title_short"] = top_vids["title"].apply(lambda t: t[:20] + "…" if isinstance(t, str) and len(t) > 20 else t)
                    fig_vids = px.bar(top_vids, x="commentCount", y="title_short", orientation="h", text="commentCount")
                    st.plotly_chart(fig_vids, use_container_width=True)
        with col4:
            with st.container(border=True):
                st.markdown("**④ 작성자 활동량 Top10**")
                ta = top_authors_from_file(comments_csv_path, topn=10)
                if ta is not None and not ta.empty:
                    fig_auth = px.bar(ta, x="count", y="author", orientation="h", text="count")
                    st.plotly_chart(fig_auth, use_container_width=True)
                else:
                    st.info("작성자 데이터 없음")

    with st.container(border=True):
        st.markdown("**⑤ 좋아요 Top10 댓글**")
        best = []
        for chunk in pd.read_csv(comments_csv_path, usecols=["video_id","video_title","author","text","likeCount"], chunksize=200_000):
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

# ============== 7) 상태 초기화 ==============
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.last_schema = None
    st.session_state.last_csv = ""
    st.session_state.last_df = None
    st.session_state.sample_text = ""
    st.session_state.messages.append({
        "role":"assistant",
        "content": "원하는 분석을 한 줄로 말해줘. 예) `최근 12시간 태풍상사 김준호 댓글반응 분석해줘`"
    })

# ============== 8) 채팅 렌더링 ==============
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ============== 9) 입력 핸들러 ==============
prompt = st.chat_input(placeholder="한 줄로 요청하세요 (예: 최근 24시간 폭군의셰프 반응 분석)")
if prompt:
    # 사용자 메시지 추가
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # 후속질문 여부
    followup = bool(st.session_state.last_csv)
    phase = st.progress(0.0, text=("해석중…" if not followup else "맥락 구성중…"))

    # 첫 질문: 파이프라인 전체 실행
    if not followup:
        # 1) 해석
        if not GEMINI_API_KEYS:
            msg = "Gemini API Key가 없습니다. st.secrets에 `GEMINI_API_KEYS` 또는 `GEMINI_API_KEY`를 설정하세요."
            st.session_state.messages.append({"role":"assistant","content":msg}); safe_rerun()
        light = call_gemini_text("", LIGHT_PROMPT.replace("{USER_QUERY}", prompt))
        schema = parse_light_block_to_schema(light)
        st.session_state.last_schema = schema
        phase.progress(0.10, text="영상 검색중…")

        # 2) YouTube 검색/통계
        if not YT_API_KEYS:
            msg = "YouTube API Key가 없습니다. st.secrets에 `YT_API_KEYS`를 설정하세요."
            st.session_state.messages.append({"role":"assistant","content":msg}); safe_rerun()
        start_dt = datetime.fromisoformat(schema["start_iso"]).astimezone(KST)
        end_dt   = datetime.fromisoformat(schema["end_iso"]).astimezone(KST)
        published_after = kst_to_rfc3339_utc(start_dt)
        published_before = kst_to_rfc3339_utc(end_dt)
        keywords = schema.get("keywords", [])
        entities = schema.get("entities", [])
        include_replies = bool(schema.get("options",{}).get("include_replies", False))

        rt = RotatingYouTube(YT_API_KEYS)
        all_ids = []
        for base_kw in (keywords or ["유튜브"]):
            all_ids += yt_search_videos(rt, base_kw, 60, "relevance", published_after, published_before)
            for e in (entities or []):
                all_ids += yt_search_videos(rt, f"{base_kw} {e}", 30, "relevance", published_after, published_before)
        all_ids = list(dict.fromkeys(all_ids))
        df_stats = pd.DataFrame(yt_video_statistics(rt, all_ids))
        if not df_stats.empty and "publishedAt" in df_stats.columns:
            df_stats["publishedAt_kst"] = (
                pd.to_datetime(df_stats["publishedAt"], errors="coerce", utc=True)
                .dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M:%S")
            )
        st.session_state.last_df = df_stats
        phase.progress(0.35, text="댓글 수집중…")

        # 3) 댓글 수집
        video_list = df_stats.to_dict('records') if not df_stats.empty else []
        csv_path, total_cnt = parallel_collect_comments_streaming(
            video_list=video_list,
            rt_keys=YT_API_KEYS,
            include_replies=include_replies,
            max_total_comments=MAX_TOTAL_COMMENTS,
            max_per_video=MAX_COMMENTS_PER_VIDEO,
            prog=phase
        )
        st.session_state.last_csv = csv_path
        if total_cnt == 0:
            st.session_state.messages.append({"role":"assistant","content":"수집된 댓글이 없습니다. 기간/키워드를 조정해줘."})
            safe_rerun()
        phase.progress(0.90, text="AI 분석중…")

        # 4) AI 요약 (첫 답변)
        sample_text, _, _ = serialize_comments_for_llm_from_file(csv_path)
        st.session_state.sample_text = sample_text

        sys = ("너는 유튜브 댓글을 분석하는 어시스턴트다. "
               "아래 키워드/엔티티와 지정된 기간의 댓글 샘플을 바탕으로 핵심 포인트를 항목화하고, "
               "긍/부/중 비율과 대표 코멘트(10개 미만)를 제시하라.")
        payload = (
            f"[키워드]: {', '.join(keywords)}\n"
            f"[엔티티]: {', '.join(entities)}\n"
            f"[기간(KST)]: {schema['start_iso']} ~ {schema['end_iso']}\n\n"
            f"[댓글 샘플]:\n{sample_text}\n"
        )
        answer_md = call_gemini_text(sys, payload)
        meta = f"*분석키워드: {', '.join(keywords) or '(없음)'} · 기간: {schema['start_iso']} ~ {schema['end_iso']}*"
        full = meta + "\n\n" + (answer_md or "_빈 응답_")
        st.session_state.messages.append({"role":"assistant","content":full})
        phase.progress(1.0, text="완료")
        safe_rerun()

    # 후속 질문: 재수집 없이 답변만
    else:
        schema = st.session_state.last_schema or {}
        sample_text = st.session_state.sample_text or ""
        # 최근 대화 10개 맥락
        lines = []
        for m in st.session_state.messages[-10:]:
            if m["role"] == "user": lines.append(f"[이전 Q]: {m['content']}")
            else:                   lines.append(f"[이전 A]: {m['content']}")
        context = "\n".join(lines)

        sys = ("너는 유튜브 댓글을 분석하는 어시스턴트다. "
               "아래 직렬화된 댓글 샘플(고정)과 이전 대화 맥락을 바탕으로, 현재 질문에 간결하고 구조화된 답을 하라. "
               "반드시 댓글 샘플에 근거하고, 인용 예시는 5개 이하로 제한하라.")
        payload = (
            context + "\n\n" +
            f"[현재 질문]: {prompt}\n"
            f"[기간(KST)]: {schema.get('start_iso','?')} ~ {schema.get('end_iso','?')}\n\n"
            f"[댓글 샘플]:\n{sample_text}\n"
        )
        answer_md = call_gemini_text(sys, payload)
        meta = f"*분석키워드: {', '.join(schema.get('keywords', [])) or '(없음)'} · 기간: {schema.get('start_iso','?')} ~ {schema.get('end_iso','?')}*"
        full = meta + "\n\n" + (answer_md or "_빈 응답_")
        st.session_state.messages.append({"role":"assistant","content":full})
        phase.progress(1.0, text="완료")
        safe_rerun()

# ============== 10) 정량 + 다운로드 (항상 하단) ==============
if st.session_state.last_csv:
    render_quant_viz_from_paths(
        st.session_state.last_csv,
        st.session_state.last_df,
        query_kw=", ".join((st.session_state.last_schema or {}).get("keywords", [])),
    )
    st.markdown("---")
    with open(st.session_state.last_csv, "rb") as f:
        st.download_button("⬇️ 전체 댓글 CSV", data=f.read(),
                           file_name=f"chatbot_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")
    if st.session_state.last_df is not None and not st.session_state.last_df.empty:
        csv_videos = st.session_state.last_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("⬇️ 전체 영상목록 CSV", data=csv_videos,
                           file_name=f"chatbot_videos_{len(st.session_state.last_df)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")

# ============== 11) 사이드 상태/도구 ==============
with st.sidebar:
    st.header("⚙️ 상태")
    st.caption("개발/진단용")
    st.subheader("키 상태")
    st.write(f"YT Keys: {len(YT_API_KEYS)}개 / Gemini Keys: {len(GEMINI_API_KEYS)}개")
    st.subheader("분석 기간/키워드")
    if st.session_state.last_schema:
        s = st.session_state.last_schema
        st.markdown(f"- 기간: {s['start_iso']} ~ {s['end_iso']}")
        st.markdown(f"- 키워드: {', '.join(s.get('keywords', [])) or '(없음)'}")
        if s.get("entities"): st.markdown(f"- 보조: {', '.join(s.get('entities', []))}")
    else:
        st.info("아직 없음")

    st.subheader("수집된 댓글 수")
    if st.session_state.last_csv and os.path.exists(st.session_state.last_csv):
        try:
            cnt = sum(1 for _ in open(st.session_state.last_csv, "r", encoding="utf-8-sig")) - 1
            st.info(f"{max(cnt,0):,}개 (추정)")
        except Exception:
            st.info("집계 불가")
    else:
        st.info("없음")

    st.markdown("---")
    if st.button("🗑️ 세션 초기화", type="secondary"):
        st.session_state.clear(); safe_rerun()
