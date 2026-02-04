# content_generator_pipeline/reddit.py
from __future__ import annotations

from typing import List, Dict
import re
import html
import xml.etree.ElementTree as ET

import requests


_IMG_RE = re.compile(r'<img[^>]+src="([^"]+)"', re.IGNORECASE)


def _extract_first_img_from_html(raw: str) -> str:
    """
    Reddit Atom content/summary often contains escaped HTML like:
      &lt;img src="..."&gt;
    So we must unescape first.
    """
    if not raw:
        return ""

    # unescape HTML entities (&lt;img ...&gt; -> <img ...>)
    s = html.unescape(raw)

    m = _IMG_RE.search(s)
    if not m:
        return ""
    url = m.group(1).replace("&amp;", "&").strip()
    return url


def _safe_image(url: str) -> str:
    """
    Safe-ish allowlist:
    - i.redd.it (direct reddit images)
    - preview.redd.it (reddit-generated previews, usually OG/preview)
    """
    if not isinstance(url, str) or not url:
        return ""
    u = url.strip()
    if "i.redd.it/" in u:
        return u
    if "preview.redd.it/" in u:
        return u
    return ""


def fetch_rss_entries(rss_url: str, max_items: int = 25) -> List[Dict]:
    """
    Fetch Reddit RSS feed and return list of entries with keys:
    - title
    - link
    - summary
    - published
    - rss
    - hero_image (optional)
    - hero_image_kind (optional)
    """
    r = requests.get(
        rss_url,
        timeout=25,
        headers={"User-Agent": "Mozilla/5.0 (Content GeneratorBot/1.0)"},
    )
    r.raise_for_status()

    root = ET.fromstring(r.text)

    ns = {
        "a": "http://www.w3.org/2005/Atom",
        "m": "http://search.yahoo.com/mrss/",
        "c": "http://purl.org/rss/1.0/modules/content/",
    }

    entries: List[Dict] = []

    for ent in root.findall("a:entry", ns):
        title_el = ent.find("a:title", ns)
        link_el = ent.find("a:link", ns)
        summary_el = ent.find("a:summary", ns)
        published_el = ent.find("a:published", ns)
        content_el = ent.find("a:content", ns)

        title = (title_el.text or "").strip() if title_el is not None else ""

        link = ""
        if link_el is not None:
            link = (link_el.attrib.get("href") or "").strip()

        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        published = (published_el.text or "").strip() if published_el is not None else ""

        content_html = (content_el.text or "").strip() if content_el is not None else ""

        # Try content -> summary -> media:thumbnail
        img = _extract_first_img_from_html(content_html) or _extract_first_img_from_html(summary)

        if not img:
            # Sometimes Atom includes media:thumbnail
            thumb_el = ent.find("m:thumbnail", ns)
            if thumb_el is not None:
                img = (thumb_el.attrib.get("url") or "").strip()

        img = _safe_image(img)

        if not title or not link:
            continue

        entries.append(
            {
                "title": title,
                "link": link,
                "summary": summary,
                "published": published,
                "rss": rss_url,
                "hero_image": img,
                "hero_image_kind": ("reddit_image" if "i.redd.it/" in img else ("reddit_preview" if "preview.redd.it/" in img else "none")),
            }
        )

        if len(entries) >= max_items:
            break

    return entries
