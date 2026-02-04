# content_generator_pipeline/generate.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from slugify import slugify
import json
import random
import re
import html as _html
import urllib.request
from urllib.parse import urlparse

from content_generator_pipeline.util import (
    ROOT,
    read_text,
    write_text,
    read_json,
    write_json,
    normalize_url,
    simple_tokens,
    jaccard,
    sanitize_llm_html,
)
from content_generator_pipeline.deepseek import DeepSeekClient
from content_generator_pipeline.reddit import fetch_rss_entries
from content_generator_pipeline.render import env_for, render_to_file, write_asset

CONFIG_PATH = ROOT / "content_generator_pipeline" / "config.json"
ADS_JSON_PATH = ROOT / "content_generator_pipeline" / "ads.json"
AD_STATE_PATH = ROOT / "data" / "ad_state.json"
PROCESSED_PATH = ROOT / "processed_urls.txt"
ARTICLES_PATH = ROOT / "data" / "articles.json"
LAST_RUN_PATH = ROOT / "data" / "last_run.json"
SITE_DIR = ROOT

TEMPLATES_DIR = ROOT / "content_generator_pipeline" / "templates"
STATIC_DIR = ROOT / "content_generator_pipeline" / "static"

# === Ads (strings must be standalone and syntactically valid) ===
ADS_TOP = "".strip()

ADS_MID = ADS_TOP
ADS_BOTTOM = ADS_TOP

ads_rail_left = """
<script src="https://pl28593834.effectivegatecpm.com/bf/0c/41/bf0c417e61a02af02bb4fab871651c1b.js"></script>
""".strip()

ads_rail_right = """
<script src="https://quge5.com/88/tag.min.js" data-zone="206389" async data-cfasync="false"></script>
""".strip()

FIXED_POLICY_BLOCK = """
<p><strong>Policy & Transparency (to stay search-friendly)</strong></p>
<ul>
  <li><strong>Source & attribution:</strong> Each post is based on a public Reddit RSS item. We always link to the original Reddit post and do not claim ownership of third-party content.</li>
  <li><strong>Original value:</strong> We add commentary, context, and takeaways. If something is uncertain, we label it as speculation rather than stating it as fact.</li>
  <li><strong>No manipulation:</strong> No cloaking, hidden text, doorway pages, or misleading metadata. Titles and summaries reflect the on-page content.</li>
  <li><strong>Safety filters:</strong> We skip obvious adult/self-harm/gore keywords and avoid NSFW feeds.</li>
  <li><strong>Ads:</strong> Third-party scripts may show ads we do not directly control. If you see problematic ads, contact us and we will adjust providers/placement.</li>
  <li><strong>Removal requests:</strong> If you believe content should be removed (copyright, personal data, etc.), email us with the URL and justification.</li>
</ul>
<p>Contact: <a href="mailto:{contact_email}">{contact_email}</a></p>
""".strip()

def load_ads_catalog() -> dict:
    """
    Load affiliate ads from content_generator_pipeline/ads.json.
    You (the user) only copy-paste codes into this JSON.
    """
    if not ADS_JSON_PATH.exists():
        return {"general": []}
    try:
        return json.loads(ADS_JSON_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to load ads.json: {e}")


def load_ad_state() -> dict:
    if not AD_STATE_PATH.exists():
        return {"shown": {}, "clicks": {}}
    try:
        return json.loads(AD_STATE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to load ad_state.json: {e}")


def save_ad_state(state: dict) -> None:
    AD_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    AD_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def classify_genre(title: str, summary: str) -> str:
    """
    Phase1: simple keyword rules (fast + stable).
    Fallback: 'general'
    """
    text = f"{title} {summary}".lower()

    rules = [
        ("health", ["health", "hair", "sleep", "diet", "doctor", "study says", "medical", "wellness"]),
        ("beauty", ["skincare", "beauty", "cosmetic", "laser", "dermatology", "makeup"]),
        ("finance", ["stock", "crypto", "bitcoin", "bank", "interest rate", "loan", "tax", "investment"]),
        ("tech", ["ai", "openai", "model", "gpu", "software", "bug", "security", "iphone", "android"]),
        ("travel", ["travel", "flight", "hotel", "trip", "tourism", "airport"]),
        ("study", ["learn", "exam", "toeic", "eiken", "study", "university"]),
        ("vpn", ["vpn", "privacy", "proxy", "geoblock"]),
        ("tools", ["tool", "formatter", "converter", "generator", "app", "extension"])
    ]

    for genre, kws in rules:
        if any(k in text for k in kws):
            return genre

    return "general"

ADS_PATH = None  # set below in main init if you already have ROOT; otherwise leave

def load_ads(ads_path):
    # ads.json format: { "genre": [ {id,title,code,detail}, ... ], ... }
    try:
        p = Path(ads_path)
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ads] load failed: {e}")
        return {}

def pick_ad_for_genre(ads_dict: dict, genre: str):
    # returns a single ad dict or None
    if not isinstance(ads_dict, dict):
        return None

    pool = ads_dict.get(genre) or []

    if not pool:
        # fallback: pick from all genres (NO 'general')
        all_ads = []
        for k, v in ads_dict.items():
            if k == "general":
                continue
            if isinstance(v, list):
                all_ads.extend([x for x in v if isinstance(x, dict)])
        pool = all_ads

    if not pool:
        return None

    try:
        return random.choice(pool)
    except Exception:
        return pool[0]


def render_affiliate_section(ad: dict) -> str:
    """
    Build a small HTML block to append at the end of body_html.
    ad['code'] is pasted by the user as-is (HTML/JS/link).
    """
    if not ad:
        return ""
    title = (ad.get("title") or "Recommended").strip()
    detail = (ad.get("detail") or "").strip()
    code = (ad.get("code") or "").strip()
    if not code:
        return ""

    # Keep tags simple (<p>, <h2>, <ul><li>, <a>) to survive sanitizer.
    parts = []
    parts.append("<h2>Recommended</h2>")
    if detail:
        parts.append(f"<p>{html.escape(detail)}</p>")
    # IMPORTANT: code is inserted raw (user-provided). Do NOT escape.
    parts.append(f"<p>{code}</p>")

    return "\n".join(parts).strip()
RELATED_GENRES: dict[str, list[str]] = {
    # tech/news 系の近似
    "tech": ["productivity", "education", "business"],
    "ai": ["tech", "education", "productivity"],
    "business": ["productivity", "tech", "education"],

    # インフラ・住環境系の近似
    "home_improvement": ["health", "education", "tech"],
    "energy": ["home_improvement", "tech", "business"],

    # 健康・美容系の近似
    "health": ["home_improvement", "education", "productivity"],
    "beauty": ["health", "home_improvement", "productivity"],

    # 旅行・ライフ系の近似
    "travel": ["productivity", "home_improvement", "business"],
}

def choose_ad(ads_catalog: dict, genre: str) -> tuple[dict | None, str | None]:
    """
    Pick the closest possible ad by genre.
    - Never use 'general'
    - Prefer exact genre pool
    - If empty, try RELATED_GENRES[genre] pools in order
    - If still empty, pick random from ALL non-general ads
    Returns: (ad_dict or None, picked_genre or None)
    """
    if not isinstance(ads_catalog, dict):
        return (None, None)

    def pool_for(g: str) -> list[dict]:
        v = ads_catalog.get(g) or []
        if not isinstance(v, list):
            return []
        return [x for x in v if isinstance(x, dict) and str(x.get("code", "")).strip()]

    # 1) exact
    if genre != "general":
        pool = pool_for(genre)
        if pool:
            return (random.choice(pool), genre)

    # 2) related genres
    for g in RELATED_GENRES.get(genre, []):
        if g == "general":
            continue
        pool = pool_for(g)
        if pool:
            return (random.choice(pool), g)

    # 3) last resort: any non-general
    all_ads: list[dict] = []
    for k, v in ads_catalog.items():
        if k == "general":
            continue
        if isinstance(v, list):
            all_ads.extend([x for x in v if isinstance(x, dict) and str(x.get("code", "")).strip()])

    if not all_ads:
        return (None, None)

    return (random.choice(all_ads), None)




def build_affiliate_section(article_id: str, title: str, summary: str, ads_catalog: dict, base_url: str) -> tuple[str, str | None]:
    """
    Returns (html, chosen_ad_id)
    Uses Cloudflare Worker /go for click tracking and redirect.
    """
    genre = classify_genre(title, summary)
    ad, picked_genre = choose_ad(ads_catalog, genre)

    if not ad:
        return ("", None)

    ad_id = str(ad.get("id", "")).strip() or None
    raw_code = str(ad.get("code", "")).strip()

    if not raw_code:
        return ("", None)

    # Click tracking URL (Cloudflare Worker handles /go)
    # Example: https://content_generator.mikanntool.com/go?ad=health-001&a=2026-01-31-xxxx
    go_url = f"{base_url.rstrip('/')}/go?ad={_html.escape(ad_id or 'unknown')}&a={_html.escape(article_id)}"

    # We do NOT parse or modify your affiliate code.
    # We only wrap it in a section and add a tracked button.
    html = f"""
<section class="card affiliate">
  <div class="card-h">
    <div><strong>Recommended (matched to this story)</strong></div>
    <div class="muted">Category: {genre}</div>
  </div>

  

  <div class="ad-slot ad-affiliate">
    {raw_code}
  </div>

  <div class="cta-row">
    <a class="pill" href="{go_url}" rel="nofollow sponsored noopener" target="_blank">
      View offer
    </a>
  </div>
</section>
""".strip()

    return (html, ad_id)

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_processed() -> set[str]:
    s = read_text(PROCESSED_PATH)
    lines = [normalize_url(x) for x in s.splitlines() if x.strip()]
    return set(lines)


def append_processed(url: str) -> None:
    url = normalize_url(url)
    existing = load_processed()
    if url in existing:
        return

    current = read_text(PROCESSED_PATH).rstrip()
    if current.strip():
        current += "\n"
    current += url + "\n"
    write_text(PROCESSED_PATH, current)

def og_image_from_article(base_url: str, a: dict) -> str:
    img = (a.get("hero_image") or "").strip()
    if not img:
        return ""

    if img.startswith("//"):
        return "https:" + img
    if img.startswith("/"):
        return base_url.rstrip("/") + img
    if img.startswith("http://") or img.startswith("https://"):
        return img

    return ""

def is_blocked(title: str, blocked_kw: list[str]) -> bool:
    t = (title or "").lower()
    for kw in blocked_kw:
        if kw.lower() in t:
            return True
    return False


def pick_candidate(cfg: dict, processed: set[str], articles: list[dict]) -> dict | None:
    blocked_kw = cfg["safety"]["blocked_keywords"]

    prev_titles = [a.get("title", "") for a in articles]
    prev_tok = [simple_tokens(t) for t in prev_titles if t]

    candidates: list[dict] = []
    for rss in cfg["feeds"]["reddit_rss"]:
        for e in fetch_rss_entries(rss):
            link = normalize_url(e["link"])
            if not link or link in processed:
                continue

            if is_blocked(e["title"], blocked_kw):
                continue

            tok = simple_tokens(e["title"])
            too_similar = any(jaccard(tok, pt) >= 0.78 for pt in prev_tok)
            if too_similar:
                continue

            # ✅ RSSから拾った安全な画像だけ使う（i.redd.itのみ）
            e["image_url"] = e.get("hero_image", "") or ""
            e["image_kind"] = e.get("hero_image_kind", "none") or "none"

            candidates.append(e)

    return candidates[0] if candidates else None


def deepseek_article(cfg: dict, item: dict) -> tuple[str, str]:

    ds = DeepSeekClient()
    model = cfg["generation"]["model"]
    target_words = int(cfg["generation"]["target_words"])
    temp = float(cfg["generation"]["temperature"])

    title = item["title"]
    link = item["link"]
    summary = item.get("summary", "")

    system = (
        "You are a high-performance conversion copywriter and tech analyst. "
        "Your mission is to write content that grabs attention, triggers the reader's survival instinct (FOMO), "
        "and provides immediate, actionable solutions. "
        "Write in English only. Do not fabricate facts. "
        "Be punchy, direct, and slightly provocative to drive clicks, but remain ethically grounded. "
        "Your goal is to make the reader feel that ignoring this info is a mistake. "
        "Always focus on the 'What's in it for me?' for the reader."
    )

    user = f"""
You are an expert tech journalist and high-conversion copywriter.
Your goal is NOT to summarize the news. Your goal is: Create an URGENT "reader-benefit" briefing that makes the reader feel, "If I don't read this now, I'm losing money/security/time."

OUTPUT RULES:
VERY IMPORTANT OUTPUT FORMAT:
- First line MUST be: TITLE: <your best SEO-friendly title>
- Second line MUST be empty (blank line).
- From the third line, output the HTML body only (allowed tags only).

- English only.
- HTML body only.
- Allowed tags: <p>, <h2>, <ul>, <li>, <strong>, <code>, <a>
- Do NOT output <h1>.
- Do NOT repeat the post title in the body.
- Do NOT paste any affiliate code or scripts.
- Do NOT invent facts. If unknown, explicitly say "Not stated in the source."

INPUT:
Post title: {title}
Permalink: {link}
RSS summary snippet (may be partial): {summary}

TASK (Mental preparation):
1) Identify the Persona: Who stands to lose the MOST (money, data, or reputation) from this news?
2) Identify the Pain: What is the single most terrifying or frustrating consequence for them?
3) Identify the Gain: What is the "unfair advantage" they get by knowing this 5 minutes before others?

IF YOU CANNOT GIVE CLEAR ACTIONS (Irrelevant/Low-value news):
- Start with: <p><strong>[SKIP: no actionable value]</strong></p>
- Then add ONE short <p> explaining why (missing specifics, no impact, etc.).
- Stop.

OUTPUT STRUCTURE (Follow this EXACTLY for maximum impact):

1) <p><strong>[CRITICAL SUMMARY]</strong>: <strong>2 lines of high-impact warning.</strong> Who is in immediate danger/losing out, and the single most urgent action to take right now.</p>

2) <h2>Is this your problem?</h2>
   <p>Check if you are in the "Danger Zone":</p>
   <ul><li>5 yes/no conditions that describe the reader's current setup or behavior. Make them feel "This is about ME."</li></ul>

3) <h2>The Hidden Reality</h2>
   <p>Summarize what changed, but focus on the <strong>IMPACT</strong>. Why does this matter more than people think? (2-3 sentences max, no fluff).</p>

4) <h2>Stop the Damage / Secure the Win</h2>
   <ul><li>3-7 concrete, actionable steps. Use strong verbs (e.g., "Revoke," "Switch," "Deploy"). If information is missing, state what to watch out for.</li></ul>

5) <h2>The High Cost of Doing Nothing</h2>
   <p>Explain the exact negative outcome (data loss, wasted cash, missed opportunity) in vivid detail. Be direct and blunt.</p>

6) <h2>Common Misconceptions</h2>
   <ul><li>3-5 "dangerous myths" about this news that will cause people to fail.</li></ul>

7) <h2>Critical FAQ</h2>
   <ul><li>5 high-stakes questions the reader is likely panicking about. If not in source, answer: "Not stated in the source."</li></ul>

8) <h2>Verify Original Details</h2>
   <p><a href="{link}" rel="nofollow noopener" target="_blank">Access the full source here</a></p>

9) <h2>Strategic Next Step</h2>
   <p>
   Write EXACTLY one transition paragraph (2–4 sentences) that bridges the current problem to a broader solution.
   - Tone: Helpful, authoritative, and practical. 
   - Strategy: "Since this news shows how vulnerable [Category] is, the smart long-term move is to [Related Best Practice]."
   - NOT mention discounts, coupons, promo codes, prices, or "buy now."
   - End with: "If you want a practical option people often use to handle this, here’s one."
   Use the Ad context below for relevance.
   </p>

AD CONTEXT (DO NOT SELL; only allow a neutral transition):
- Genre: {{AD_GENRE}}
- Ad title: {{AD_TITLE}}
- Ad detail: {{AD_DETAIL}}

FINAL TOUCH:
- Put it as the very last paragraph, max 2 sentences. Focus on "Choosing trusted standards/tools" in this domain to avoid scams or repeat issues.
""".strip()

    # ---- Phase1: pick one affiliate ad by genre and feed details to the prompt ----
    # Decide genre from title/summary (your classify_genre already exists)
    genre = classify_genre(title, summary)

    # Resolve ads.json path (keep your current location: content_generator_pipeline/ads.json)
    # If you already have ROOT defined earlier, use it; otherwise fall back to relative.
    ads_path = str(Path("content_generator_pipeline") / "ads.json")
    ads_dict = load_ads(ads_path)
    ad = pick_ad_for_genre(ads_dict, genre)

    ad_title = (ad.get("title") if ad else "") or ""
    ad_detail = (ad.get("detail") if ad else "") or ""
    # Replace placeholders in the user prompt
    user = (user
        .replace("{{AD_GENRE}}", genre)
        .replace("{{AD_TITLE}}", ad_title)
        .replace("{{AD_DETAIL}}", ad_detail))


    out = ds.chat(
    model=model,
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ],
    temperature=temp,
    max_tokens=2400,
)

# ---- Make it robust: out can be None/empty ----
    out = ds.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temp,
        max_tokens=2400,
    )

    # ---- Make it robust: out can be None/empty ----
    out = (out or "").strip()

    # Expect:
    # TITLE: ...
    #
    # <p>...</p>...
    m = re.match(r"(?is)^\s*TITLE:\s*(.+?)\s*\n\s*\n(.*)$", out)
    if m:
        llm_title = m.group(1).strip()
        llm_html = m.group(2).strip()
    else:
        # Fallback if DeepSeek didn't follow format
        llm_title = (item.get("title") or "").strip()
        llm_html = out

    llm_html = sanitize_llm_html(llm_html or "")
    return (llm_title, llm_html)




def strip_leading_duplicate_title(body_html: str, title: str) -> str:
    if not body_html or not title:
        return body_html

    t = _html.unescape(title).strip()
    t_norm = re.sub(r"\s+", " ", t).lower()

    def _same(text: str) -> bool:
        x = _html.unescape(text or "").strip()
        x = re.sub(r"\s+", " ", x).lower()
        return x == t_norm

    s = body_html.lstrip()

    m = re.match(r"(?is)^\s*<h1[^>]*>(.*?)</h1>\s*", s)
    if m and _same(m.group(1)):
        return s[m.end() :].lstrip()

    m = re.match(r"(?is)^\s*<h2[^>]*>(.*?)</h2>\s*", s)
    if m and _same(m.group(1)):
        return s[m.end() :].lstrip()

    m = re.match(r"(?is)^\s*<p[^>]*>(.*?)</p>\s*", s)
    if m:
        inner = re.sub(r"(?is)<[^>]+>", "", m.group(1))
        if _same(inner):
            return s[m.end() :].lstrip()

    m = re.match(r"(?is)^\s*([^<\n]{10,200})\s*(?:<br\s*/?>|\n)\s*", s)
    if m and _same(m.group(1)):
        return s[m.end() :].lstrip()

    return body_html


def compute_rankings(articles: list[dict]) -> list[dict]:
    return sorted(articles, key=lambda a: a.get("published_ts", ""), reverse=True)


def related_articles(current: dict, articles: list[dict], k: int = 6) -> list[dict]:
    cur_tok = simple_tokens(current.get("title", ""))
    scored: list[tuple[float, dict]] = []
    for a in articles:
        if a.get("id") == current.get("id"):
            continue
        sim = jaccard(cur_tok, simple_tokens(a.get("title", "")))
        scored.append((sim, a))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for s, a in scored[:k] if s > 0.05]


def write_rss_feed(cfg: dict, articles: list[dict], limit: int = 10) -> None:
    base_url = cfg["site"]["base_url"].rstrip("/")
    site_title = cfg["site"].get("title", "Content Generator")
    site_desc = cfg["site"].get("description", "Daily digest")

    items = sorted(articles, key=lambda a: a.get("published_ts", ""), reverse=True)[:limit]

    def rfc822(iso: str) -> str:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")

    parts = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append("<rss version='2.0' xmlns:atom='http://www.w3.org/2005/Atom'>")
    parts.append("<channel>")
    parts.append(f"<title>{_html.escape(site_title)}</title>")
    parts.append(f"<link>{_html.escape(base_url + '/')}</link>")
    parts.append(f"<description>{_html.escape(site_desc)}</description>")
    parts.append(f"<lastBuildDate>{_html.escape(rfc822(now_utc_iso()))}</lastBuildDate>")

    for a in items:
        url = f"{base_url}{a['path']}"
        title = a.get("title", "")
        pub = a.get("published_ts", now_utc_iso())
        summary = a.get("summary", "") or ""
        if not summary:
            summary = re.sub(r"\s+", " ", re.sub(r"(?is)<[^>]+>", " ", a.get("body_html", ""))).strip()[:240]

        parts.append("<item>")
        parts.append(f"<title>{_html.escape(title)}</title>")
        parts.append(f"<link>{_html.escape(url)}</link>")
        parts.append(f"<guid isPermaLink='true'>{_html.escape(url)}</guid>")
        parts.append(f"<pubDate>{_html.escape(rfc822(pub))}</pubDate>")
        parts.append(f"<description>{_html.escape(summary)}</description>")
        parts.append("</item>")

    parts.append("</channel>")
    parts.append("</rss>")

    (SITE_DIR / "feed.xml").write_text("\n".join(parts) + "\n", encoding="utf-8")


def _abs_image_url(base_url: str, img: str) -> str:
    """og:image は絶対URLが安定。i.redd.it 等が来る想定。"""
    if not img:
        return ""
    s = img.strip()
    if s.startswith("http://") or s.startswith("https://"):
        return s
    if s.startswith("/"):
        return base_url.rstrip("/") + s
    return base_url.rstrip("/") + "/" + s

def _guess_ext_from_url(u: str) -> str:
    try:
        path = urlparse(u).path.lower()
    except Exception:
        path = ""
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if path.endswith(ext):
            return ext
    return ".jpg"


def cache_og_image(base_url: str, src_url: str, article_id: str) -> str:
    """
    RSSに画像がある記事だけ:
    # Save external images under og/ (if enabled)
    - og:image は自ドメインの絶対URLで返す
    画像が無い/失敗 → "" を返す（= og:image を出さない）
    """
    src_url = (src_url or "").strip()
    if not src_url:
        return ""

    # i.redd.it など外部からの直リンクがSNSクローラに弾かれる対策として自サイトにキャッシュ
    ext = _guess_ext_from_url(src_url)
    rel = f"/og/{article_id}{ext}"
    out_path = SITE_DIR / rel.lstrip("/")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not out_path.exists():
        try:
            req = urllib.request.Request(
                src_url,
                headers={
                    # ここ重要：UA無いと弾くCDNがある
                    "User-Agent": "Mozilla/5.0 (compatible; Content GeneratorBot/1.0; +https://content_generator.mikanntool.com/)"
                },
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            if data:
                out_path.write_bytes(data)
        except Exception:
            return ""

    # 自サイトの絶対URLを返す（SNSはこれを取りに来る）
    return base_url.rstrip("/") + rel

def build_site(cfg: dict, articles: list[dict]) -> None:
    base_url = cfg["site"]["base_url"].rstrip("/")

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "articles").mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "assets").mkdir(parents=True, exist_ok=True)

    write_asset(SITE_DIR / "assets" / "style.css", STATIC_DIR / "style.css")
    write_asset(SITE_DIR / "assets" / "fx.js", STATIC_DIR / "fx.js")

    robots = f"""User-agent: *
Allow: /

Sitemap: {base_url}/sitemap.xml
"""
    (SITE_DIR / "robots.txt").write_text(robots, encoding="utf-8")

    urls = [f"{base_url}/"] + [f"{base_url}{a['path']}" for a in articles]
    sitemap_items = "\n".join([f"<url><loc>{u}</loc></url>" for u in urls])
    sitemap = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{sitemap_items}
</urlset>
"""
    (SITE_DIR / "sitemap.xml").write_text(sitemap, encoding="utf-8")

    jenv = env_for(TEMPLATES_DIR)

    ranking = compute_rankings(articles)[:10]
    new_articles = sorted(articles, key=lambda a: a.get("published_ts", ""), reverse=True)[:10]

    write_rss_feed(cfg, articles, limit=10)

    base_ctx = {
        "site": cfg["site"],
        "ranking": ranking,
        "new_articles": new_articles,
        "ads_top": ADS_TOP,
        "ads_mid": ADS_MID,
        "ads_bottom": ADS_BOTTOM,
        "ads_rail_left": ads_rail_left,
        "ads_rail_right": ads_rail_right,
        "now_iso": now_utc_iso(),
    }

    # index.html（画像メタは出さない：デフォルト画像も出さない）
    ctx = dict(base_ctx)
    ctx.update(
        {
            "title": cfg["site"].get("title", "Content Generator"),
            "description": cfg["site"].get("description", "Daily digest"),
            "canonical": base_url + "/",
            "og_type": "website",
            "og_image": "",  # ←空なら base.html 側で出さない
        }
    )
    render_to_file(jenv, "index.html", ctx, SITE_DIR / "index.html")

    static_pages = [
        ("about", "About Content Generator", "<p>Content Generator is a daily digest that curates a single noteworthy Reddit item and adds commentary, context, and takeaways.</p>"),
        ("privacy", "Privacy", "<p>We do not require accounts. Third-party ad scripts may set cookies or collect device identifiers. See each provider’s policy. If you want removal or have concerns, contact us.</p>"),
        ("terms", "Terms", "<p>Use at your own risk. Content is informational and may be incomplete. We link to sources and welcome corrections.</p>"),
        ("disclaimer", "Disclaimer", "<p>This site is not affiliated with Reddit. Trademarks belong to their owners. We do not guarantee accuracy, availability, or outcomes.</p>"),
        ("contact", "Contact", f"<p>Email: <a href='mailto:{cfg['site']['contact_email']}'>{cfg['site']['contact_email']}</a></p>"),
    ]

    for slug, page_title, body in static_pages:
        ctx = dict(base_ctx)
        ctx.update(
            {
                "page_title": page_title,
                "page_body": body,
                "title": page_title,
                "description": cfg["site"].get("description", "Daily digest"),
                "canonical": f"{base_url}/{slug}.html",
                "og_type": "website",
                "og_image": "",  # デフォルト無し
            }
        )
        render_to_file(jenv, "static.html", ctx, SITE_DIR / f"{slug}.html")

    # 記事ページ：RSS画像がある記事だけ og:image を出す
    for a in articles:
        rel = related_articles(a, articles, k=6)

        src = a.get("hero_image", "") or ""
        og_img = cache_og_image(base_url, src, a.get("id", "article"))


        ctx = dict(base_ctx)
        ctx.update(
            {
                "a": a,
                "related": rel,
                "ranking": ranking,
                "new_articles": new_articles,
                "policy_block": FIXED_POLICY_BLOCK.format(contact_email=cfg["site"]["contact_email"]),
                "title": a.get("title", cfg["site"].get("title", "Content Generator")),
                "description": (a.get("summary", "") or cfg["site"].get("description", "Daily digest"))[:200],
                "canonical": f"{base_url}{a['path']}",
                "og_type": "article",
                "og_image": og_img,  # ←ここが空ならメタは出ない（デフォルト無し）
            }
        )
        render_to_file(jenv, "article.html", ctx, SITE_DIR / a["path"].lstrip("/"))


def write_last_run(cfg: dict, payload: dict[str, Any]) -> None:
    base_url = cfg["site"]["base_url"].rstrip("/")
    out = {
        "updated_utc": now_utc_iso(),
        "homepage_url": base_url + "/",
        **payload,
    }
    write_json(LAST_RUN_PATH, out)


def main() -> None:
    cfg = load_config()
    base_url = cfg["site"]["base_url"].rstrip("/")

    processed = load_processed()
    articles = read_json(ARTICLES_PATH, default=[])

    cand = pick_candidate(cfg, processed, articles)
    if not cand:
        build_site(cfg, articles)
        write_last_run(
            cfg,
            {
                "created": False,
                "article_url": "",
                "article_title": "",
                "source_url": "",
                "note": "No new candidate found. Site rebuilt.",
            },
        )
        return

    llm_title, body_html = deepseek_article(cfg, cand)
    body_html = strip_leading_duplicate_title(body_html, llm_title or cand["title"])


    ads_catalog = load_ads_catalog()
    affiliate_html, chosen_ad_id = build_affiliate_section(
        article_id=f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}-{slugify(llm_title or cand['title'])[:80] or 'post'}",
        title=llm_title or cand["title"],
        summary=cand.get("summary", "") or "",
        ads_catalog=ads_catalog,
        base_url=base_url,
    )

    print(f"[ads] chosen_ad_id={chosen_ad_id} affiliate_len={len(affiliate_html or '')}")

    # Append affiliate section at the end of the article body (phase1)
    if affiliate_html:
        body_html = body_html.rstrip() + "\n\n" + affiliate_html + "\n"

    ts = datetime.now(timezone.utc)

    ymd = ts.strftime("%Y-%m-%d")
    slug =slugify(llm_title or cand['title'])[:80] or f"post-{int(ts.timestamp())}"
    path = f"/articles/{ymd}-{slug}.html"
    article_url = base_url + path

    entry = {
        "id": f"{ymd}-{slug}",
        "title": llm_title or cand["title"],
        "path": path,
        "published_ts": ts.isoformat(timespec="seconds"),
        "source_url": cand["link"],
        "rss": cand.get("rss", ""),
        "summary": cand.get("summary", ""),
        "body_html": body_html,
        # ✅ RSSから拾った安全画像（i.redd.itのみ）。無ければ空で表示されない
        "hero_image": cand.get("image_url", "") or "",
        "hero_image_kind": cand.get("image_kind", "none") or "none",
    }

    append_processed(cand["link"])
    articles.insert(0, entry)
    write_json(ARTICLES_PATH, articles)

    build_site(cfg, articles)

    write_last_run(
        cfg,
        {
            "created": True,
            "article_url": article_url,
            "article_path": path,
            "article_title": cand["title"],
            "source_url": cand["link"],
        },
    )


if __name__ == "__main__":
    main()
