import re
import json
import logging
from urllib.parse import urlparse, parse_qs

import requests
from youtube_transcript_api import YouTubeTranscriptApi

# -----------------------------------------------------------
# LOGGER (silent by default; Streamlit can show via st.write)
# -----------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LANG_PRIORITY = ["en", "en-US", "en-GB", "hi", "hi-IN"]
VIDEO_ID_RE = re.compile(r"^[0-9A-Za-z_-]{11}$")


# -----------------------------------------------------------
# Extract YouTube video ID robustly
# -----------------------------------------------------------
def extract_video_id(url: str) -> str:
    """
    Extracts the 11-char YouTube video ID.
    Handles: youtube.com/watch?v=..., youtu.be/..., /shorts/..., /embed/...,
    /live/..., m.youtube.com, or a raw video ID.
    Raises ValueError if no valid ID can be found.
    """
    url = url.strip()
    if VIDEO_ID_RE.match(url):
        return url

    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    candidate = None
    if host.endswith("youtu.be"):
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif "youtube" in host:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            candidate = qs["v"][0]
        else:
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
                candidate = parts[1]

    if candidate and VIDEO_ID_RE.match(candidate):
        return candidate
    raise ValueError(f"Could not find a valid YouTube video ID in: {url}")


# -----------------------------------------------------------
# Fetch transcript (official API + yt-dlp fallback)
# -----------------------------------------------------------
def _fetch_with_transcript_api(video_id: str):
    """Use youtube-transcript-api (>= 1.x instance API)."""
    transcript_list = YouTubeTranscriptApi().list(video_id)

    # For each language in LANG_PRIORITY, a manual transcript is preferred over an auto-generated one
    try:
        tr = transcript_list.find_transcript(LANG_PRIORITY)
    except Exception:
        # No preferred language: take whatever exists
        tr = next(iter(transcript_list))

    # The embedding model is English-only, so translate non-English transcripts when possible
    if not tr.language_code.startswith("en") and tr.is_translatable:
        try:
            data = tr.translate("en").fetch().to_raw_data()
            logger.info(f"Transcript ({tr.language_code} translated to en) · {len(data)} segments")
            return data
        except Exception as e:
            logger.warning(f"Translation to English failed, using {tr.language_code}: {e}")

    data = tr.fetch().to_raw_data()
    logger.info(f"Transcript ({tr.language_code}, generated={tr.is_generated}) · {len(data)} segments")
    return data


def _parse_vtt(txt: str):
    segs = []
    entries = re.findall(
        r"(\d{2}):(\d{2}):(\d{2})\.(\d{3}) --> [^\n]*\n(.*?)(?:\n\n|\Z)", txt, re.S
    )
    last = None
    for h, m, s, ms, body in entries:
        # Strip inline VTT tags like <00:00:01.000><c>word</c>
        line = re.sub(r"<[^>]+>", "", body).replace("\n", " ").strip()
        # Auto-captions repeat each line across consecutive cues
        if line and line != last:
            segs.append({"text": line, "start": int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000})
            last = line
    return segs


def _parse_json3(txt: str):
    segs = []
    data = json.loads(txt)
    for ev in data.get("events", []):
        if ev.get("segs"):
            text = "".join(s.get("utf8", "") for s in ev["segs"]).replace("\n", " ").strip()
            if text:
                segs.append({"text": text, "start": ev.get("tStartMs", 0) / 1000.0})
    return segs


def _fetch_with_ytdlp(video_id: str):
    """Fallback: ask yt-dlp for caption URLs, download and parse them."""
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)

    # Prefer manual subtitles, then auto captions, in preferred language order
    candidates = []
    for source in (info.get("subtitles") or {}, info.get("automatic_captions") or {}):
        for code in LANG_PRIORITY:
            if code in source:
                candidates.append((code, source[code]))
    if not candidates:
        for source in (info.get("subtitles") or {}, info.get("automatic_captions") or {}):
            for code, formats in source.items():
                if code != "live_chat":
                    candidates.append((code, formats))
    if not candidates:
        raise RuntimeError("No captions found via yt-dlp")

    code, formats = candidates[0]
    by_ext = {f.get("ext"): f for f in formats}
    for ext, parser in (("json3", _parse_json3), ("vtt", _parse_vtt)):
        if ext in by_ext:
            try:
                resp = requests.get(by_ext[ext]["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
                resp.raise_for_status()
                segs = parser(resp.text)
            except Exception as e:
                logger.warning(f"yt-dlp {ext} captions failed: {e}")
                continue
            if segs:
                logger.info(f"yt-dlp extracted {len(segs)} caption lines ({code}, {ext})")
                return segs
    raise RuntimeError("yt-dlp found captions but could not download or parse them")


def fetch_transcript(video_id: str):
    """
    Fetch YouTube transcript using multiple fallback strategies:
      1. youtube-transcript-api (manual & auto-generated captions)
      2. yt-dlp fallback (json3 / vtt captions)
    Returns: list of dicts like {"text": "...", "start": seconds}
    Raises: Exception with user-friendly message if all fail.
    """
    errors = []
    for name, fn in (("transcript-api", _fetch_with_transcript_api), ("yt-dlp", _fetch_with_ytdlp)):
        try:
            data = fn(video_id)
            if data:
                return data
        except Exception as e:
            logger.warning(f"{name} failed: {e}")
            errors.append(f"{name}: {str(e).splitlines()[0] if str(e) else type(e).__name__}")

    raise Exception(
        "No transcript available for this video. "
        "Please try a different video with captions enabled.\n\nDetails: " + " | ".join(errors)
    )


# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------
def transcript_to_text(transcript):
    """Flatten list of {text,start} into a single concatenated string."""
    return " ".join(
        seg.get("text", "").replace("\n", " ").replace("\xa0", " ").strip()
        for seg in transcript
        if seg.get("text", "").strip()
    )
