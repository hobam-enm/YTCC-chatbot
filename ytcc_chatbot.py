# -*- coding: utf-8 -*-
# 💬 유튜브 댓글분석기 — 챗봇 모드 (미니멀 UI)
# - 자연어 한 줄 → (기간/키워드/옵션) 해석(Gemini) → 수집 → 요약/시각화
# - UI는 검색바 1개 + 결과 상단에 작은 배지(분석키워드/분석기간) + 심플 로딩
# - 다운로드: 전체댓글 CSV, 전체영상목록 CSV

import streamlit as st
import pandas as pd
import os, re, json, gc, time
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai
import plotly.express as px

# ===================== 기본 설정 =====================
st.set_page_config(page_title="💬 댓글분석기: 챗봇 모드", layout="wide", initial_sidebar_state="centered")

# 헤더(스플래시 느낌)
st.markdown(
    """
    <div style="text-align:center; margin-top:24px; margin-bottom:12px">
      <div style="font-size:38px; font-weight:700; letter-spacing:-0.5px;">Streamlit AI assistant</div>
      <div style="color:#6b7280; margin-top:6px">Ask a question…</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===================== 비밀키/상수 =====================
BASE_DIR = "/tmp"; os.makedirs(BASE_DIR, exist_ok=True)
KST = timezone(timedelta(hours=9))

_YT_FALLBACK = []
_GEM_FALLBACK = []
YT_API_KEYS = list(st.secrets.get("YT_API_KEYS", [])) or _YT_FALLBACK
GEMINI_API_KEYS = list(st.secrets.get("GEMINI_API_KEYS", [])) or _GEM_FALLBACK
GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_TIMEOUT = int(st.secrets.get("GEMINI_TIMEOUT", 120))
GEMINI_MAX_TOKENS = int(st.secrets.get("GEMINI_MAX_TOKENS", 2048))

MAX_TOTAL_COMMENTS = 120_000
MAX_COMMENTS_PER_VIDEO = 4_000

# ===================== 유틸 =====================

def now_kst():
    return datetime.now(tz=KST)

def to_iso_kst(dt: datetime) -> str:
    if dt.tzinfo is None: dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST).isoformat(timespec="seconds")

def kst_to_rfc3339_utc(dt_kst: datetime) -> str:
    if dt_kst.tzinfo is None: dt_kst = dt_kst.replace(tzinfo=KST)
    return dt_kst.astimezone(timezone.utc).isoformat().replace("+00:00","Z")

class RotatingKeys:
    def __init__(self, keys, state_key: str):
        ks = [k.strip() for k in (keys or []) if isinstance(k, str) and k.strip()]
        self.keys = ks[:10]; self.state_key = state_key
        self.idx = st.session_state.get(state_key, 0) % max(1, len(self.keys))
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
        key = self.rot.current();
        if not key: raise RuntimeError("YouTube API Key가 없습니다.")
        self.svc = build("youtube", "v3", developerKey=key)
    def execute(self, factory):
        try:
            return factory(self.svc).execute()
        except HttpError as e:
            status = getattr(getattr(e, 'resp', None), 'status', None)
            msg = (getattr(e, 'content', b'').decode('utf-8', errors='ignore') or '').lower()
            if status in (403,429) and any(t in msg for t in ["quota","rate","limit"]) and len(YT_API_KEYS) > 1:
                self.rot.rotate(); self.svc = build("youtube", "v3", developerKey=self.rot.current())
                return factory(self.svc).execute()
            raise

# ===================== Gemini 호출 & 프롬프트 =====================
LIGHT_PROMPT = (
    "역할: 당신은 ‘유튜브 댓글 반응 분석기’를 위한 자연어 해석가다.\n"
    "목표: 사용자의 요청에서 [검색 기간]과 [검색 키워드(주제/엔티티/보조어)]를 해석한다.\n"
    "원칙: 기간은 KST(+09:00)로, 상대기간의 종료시점은 지금. 옵션(대댓글 포함/공식채널/언어)도 감지.\n\n"
    "출력 형식(6줄 고정):\n"
    "- 한 줄 요약: <문장>\n"
    "- 기간(KST): <YYYY-MM-DDTHH:MM:SS+09:00> ~ <YYYY-MM-DDTHH:MM:SS+09:00>\n"
    "- 키워드: [<메인1>, <메인2>…]\n"
    "- 엔티티/보조: [<보조들>]\n"
    "- 옵션: { include_replies: true|false, channel_filter: \"any|official|unofficial\", lang: \"ko|en|auto\" }\n"
    "- 원문: {USER_QUERY}\n\n"
    f"지금 시간 KST: {to_iso_kst(now_kst())}\n아래 입력 해석:\n\n{{USER_QUERY}}"
)

def call_gemini(prompt: str) -> str:
    rk = RotatingKeys(GEMINI_API_KEYS, "gem_key_idx")
    if not rk.current(): raise RuntimeError("Gemini API Key가 없습니다.")
    genai.configure(api_key=rk.current())
    model = genai.GenerativeModel(GEMINI_MODEL, generation_config={"temperature":0.2,"max_output_tokens":GEMINI_MAX_TOKENS})
    try:
        resp = model.generate_content([prompt], request_options={"timeout": GEMINI_TIMEOUT})
        return getattr(resp, "text", "") or ""
    except Exception:
        if len(GEMINI_API_KEYS) > 1:
            rk.rotate(); genai.configure(api_key=rk.current())
            resp = model.generate_content([prompt], request_options={"timeout": GEMINI_TIMEOUT})
            return getattr(resp, "text", "") or ""
        raise

# 6줄 라이트 블록 → 표준 스키마

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

# ===================== YouTube 검색/수집 =====================

def yt_search_videos(rt, keyword, max_results, order="relevance", published_after=None, published_before=None):
    vids, token = [], None
    while len(vids) < max_results:
        params = dict(q=keyword, part="id", type="video", order=order, maxResults=min(50, max_results - len(vids)))
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
        resp = rt.execute(lambda s: s.videos().list(part="statistics,snippet,contentDetails", id=",".join(batch)))
        for item in resp.get("items", []):
            stats = item.get("statistics", {}); snip = item.get("snippet", {})
            rows.append({
                "video_id": item.get("id"),
                "video_url": f"https://www.youtube.com/watch?v={item.get('id')}",
                "title": snip.get("title",""),
                "channelTitle": snip.get("channelTitle",""),
                "publishedAt": snip.get("publishedAt",""),
                "viewCount": int(stats.get("viewCount",0) or 0),
                "likeCount": int(stats.get("likeCount",0) or 0),
                "commentCount": int(stats.get("commentCount",0) or 0),
            })
        time.sleep(0.25)
    return rows

from googleapiclient.errors import HttpError

def yt_all_replies(rt, parent_id, video_id, title=""):
    replies, token = [], None
    while True:
        params = dict(part="snippet", parentId=parent_id, maxResults=100, pageToken=token, textFormat="plainText")
        try:
            resp = rt.execute(lambda s: s.comments().list(**params))
        except HttpError:
            break
        for c in resp.get("items", []):
            sn = c["snippet"]
            replies.append({
                "video_id": video_id, "video_title": title, "isReply": 1,
                "comment_id": c.get("id",""), "parent_id": parent_id,
                "author": sn.get("authorDisplayName",""),
                "text": sn.get("textDisplay","") or "",
                "publishedAt": sn.get("publishedAt",""),
                "likeCount": int(sn.get("likeCount",0) or 0),
            })
        token = resp.get("nextPageToken")
        if not token: break
        time.sleep(0.2)
    return replies


def yt_all_comments_sync(rt, video_id, title="", include_replies=True, max_per_video=None):
    rows, token = [], None
    while True:
        if max_per_video is not None and len(rows) >= max_per_video:
            return rows[:max_per_video]
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
                "video_id": video_id, "video_title": title, "isReply": 0,
                "comment_id": thread_id, "parent_id": "",
                "author": top.get("authorDisplayName",""),
                "text": top.get("textDisplay","") or "",
                "publishedAt": top.get("publishedAt",""),
                "likeCount": int(top.get("likeCount",0) or 0),
            })
            if include_replies and total_replies > 0:
                cap = None if max_per_video is None else max(0, max_per_video - len(rows))
                if cap == 0: return rows[:max_per_video]
                rows.extend(yt_all_replies(rt, thread_id, video_id, title)[:cap if cap else None])
                if max_per_video is not None and len(rows) >= max_per_video:
                    return rows[:max_per_video]
        token = resp.get("nextPageToken")
        if not token: break
        time.sleep(0.2)
    return rows


def parallel_collect_comments_streaming(video_list, rt_keys, include_replies, max_total_comments, max_per_video, log=None, prog=None):
    out_csv = os.path.join(BASE_DIR, f"collect_{uuid4().hex}.csv")
    wrote_header = False; total_written = 0
    total_videos = len(video_list); done = 0
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(yt_all_comments_sync, RotatingYouTube(rt_keys), v["video_id"], v.get("title",""), include_replies, max_per_video): v for v in video_list}
        for f in as_completed(futures):
            v = futures[f]
            try:
                comm = f.result()
                if comm:
                    dfc = pd.DataFrame(comm)
                    dfc.to_csv(out_csv, index=False, mode=("a" if wrote_header else "w"), header=(not wrote_header), encoding="utf-8-sig")
                    wrote_header = True; total_written += len(dfc)
            except Exception as e:
                if log: log(f"실패: {v.get('title','')} — {e}")
            done += 1
            if log: log(f"진행 {done}/{total_videos}")
            if prog: prog(done/total_videos)
            if total_written >= max_total_comments:
                break
    return out_csv, total_written

# LLM 직렬화(간단)

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

# ===================== 검색 입력(한줄) =====================
with st.container():
    c1, c2, c3 = st.columns([1,6,1])
    with c2:
        q = st.text_input(" ", placeholder="최근 12시간 태풍상사 김준호 댓글반응 분석해줘", label_visibility="collapsed", key="query")
    run = st.button("▶", help="실행", use_container_width=False)

# ===================== 실행 =====================
if run and q:
    if not (GEMINI_API_KEYS and YT_API_KEYS):
        st.error("API 키 설정이 필요합니다 (YT_API_KEYS / GEMINI_API_KEYS).")
    else:
        # 1) 해석
        with st.status("요청 해석 중…", expanded=False) as s:
            prompt = LIGHT_PROMPT.replace("{USER_QUERY}", q)
            light = call_gemini(prompt)
            schema = parse_light_block_to_schema(light)
            s.update(label="해석 완료", state="complete")

        # 2) 파라미터 준비
        start_dt = datetime.fromisoformat(schema["start_iso"]).astimezone(KST)
        end_dt   = datetime.fromisoformat(schema["end_iso"]).astimezone(KST)
        published_after = kst_to_rfc3339_utc(start_dt)
        published_before = kst_to_rfc3339_utc(end_dt)
        keywords = schema.get("keywords", [])
        entities = schema.get("entities", [])
        include_replies = bool(schema.get("options",{}).get("include_replies", False))

        # 3) 수집
        with st.status("영상/댓글 수집 중…", expanded=True) as s:
            rt = RotatingYouTube(YT_API_KEYS)
            ids_all = []
            for base in (keywords or ["유튜브"]):
                ids_all += yt_search_videos(rt, base, 60, "relevance", published_after, published_before)
                for e in (entities or []):
                    ids_all += yt_search_videos(rt, f"{base} {e}", 30, "relevance", published_after, published_before)
            ids_all = list(dict.fromkeys(ids_all))
            s.write(f"🎞️ 대상 영상: {len(ids_all)}")
            stats = yt_video_statistics(rt, ids_all)
            df_stats = pd.DataFrame(stats)
            if not df_stats.empty and "publishedAt" in df_stats.columns:
                df_stats["publishedAt_kst"] = pd.to_datetime(df_stats["publishedAt"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M:%S")
            prog = st.progress(0)
            csv_path, total_cnt = parallel_collect_comments_streaming(df_stats.to_dict('records'), YT_API_KEYS, include_replies, MAX_TOTAL_COMMENTS, MAX_COMMENTS_PER_VIDEO, log=s.write, prog=prog.progress)
            s.write(f"총 댓글 수집: {total_cnt:,}개")
            s.update(label="수집 완료", state="complete")

        if total_cnt == 0:
            st.warning("댓글이 수집되지 않았습니다. 기간/키워드를 조정해 보세요.")
            st.stop()

        # 4) 상단 메타 배지
        kw_badge = ", ".join(keywords) if keywords else "(없음)"
        period_badge = f"{schema['start_iso']} ~ {schema['end_iso']}"
        st.markdown(
            f"<div style='margin:8px 0 4px 0'>"
            f"<span style='background:#f3f4f6; padding:6px 10px; border-radius:999px; margin-right:6px; font-size:12px;'>분석키워드: {kw_badge}</span>"
            f"<span style='background:#f3f4f6; padding:6px 10px; border-radius:999px; font-size:12px;'>분석기간: {period_badge}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # 5) AI 요약
        st.subheader("🧠 요약")
        sample_text, _, _ = serialize_comments_for_llm_from_file(csv_path)
        genai.configure(api_key=RotatingKeys(GEMINI_API_KEYS, "gem_key_idx").current())
        model = genai.GenerativeModel(GEMINI_MODEL, generation_config={"temperature":0.2, "max_output_tokens":GEMINI_MAX_TOKENS})
        sys = (
            "너는 유튜브 댓글을 분석하는 어시스턴트다. "
            "아래 키워드/엔티티와 지정된 기간의 댓글 샘플을 바탕으로 핵심 포인트를 항목화하고, 긍/부/중 비율과 대표 코멘트(10개 미만)를 제시하라."
        )
        payload = f"[키워드]: {', '.join(keywords)}\n[엔티티]: {', '.join(entities)}\n[기간(KST)]: {schema['start_iso']} ~ {schema['end_iso']}\n\n[댓글 샘플]:\n{sample_text}\n"
        with st.status("AI 요약 생성 중…", expanded=False) as s:
            resp = model.generate_content([sys, payload], request_options={"timeout": GEMINI_TIMEOUT})
            st.markdown(getattr(resp, "text", ""))
            s.update(label="요약 완료", state="complete")

        # 6) 정량 하이라이트(심플)
        st.subheader("📊 정량 하이라이트")
        try:
            tmin=tmax=None; agg={}
            for chunk in pd.read_csv(csv_path, usecols=["publishedAt"], chunksize=200_000):
                dt = pd.to_datetime(chunk["publishedAt"], errors="coerce", utc=True)
                if dt.notna().any():
                    lo, hi = dt.min(), dt.max(); tmin = lo if (tmin is None or lo < tmin) else tmin; tmax = hi if (tmax is None or hi > tmax) else tmax
                dt = dt.dt.tz_convert("Asia/Seoul").dropna()
                if dt.empty: continue
                bucket = (dt.dt.floor("H") if (tmax and tmin and (tmax-tmin).total_seconds()/3600.0 <= 48) else dt.dt.floor("D"))
                vc = bucket.value_counts()
                for t, c in vc.items(): agg[t]=agg.get(t,0)+int(c)
            if agg:
                ts = pd.Series(agg).sort_index().rename("count").reset_index().rename(columns={"index":"bucket"})
                fig = px.line(ts, x="bucket", y="count", markers=True, title="댓글량 추이 (KST)")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"시계열 생성 불가: {e}")

        # 7) 다운로드 (전체댓글 / 전체영상목록)
        st.markdown("---")
        st.subheader("⬇️ 다운로드")
        with open(csv_path, "rb") as f:
            st.download_button("전체 댓글 (CSV)", data=f.read(), file_name=f"comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
        if not df_stats.empty:
            csv_videos = df_stats.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("전체 영상목록 (CSV)", data=csv_videos, file_name=f"videos_{len(df_stats)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

# 푸터 액션
st.markdown("<div style='margin:24px 0; color:#9ca3af; font-size:12px'>Legal disclaimer</div>", unsafe_allow_html=True)

st.markdown("---")
cols = st.columns(2)
with cols[0]:
    if st.button("🔄 초기화", type="secondary"): st.session_state.clear(); st.rerun()
with cols[1]:
    if st.button("🧹 캐시/메모리 정리"): st.cache_data.clear(); gc.collect(); st.success("정리 완료")
