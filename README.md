# Content Generator (Lite)

This ZIP generates **one new article per run** from an RSS feed and publishes it to a **GitHub Pages URL**.

## What this does
- Fetches one RSS item (example: Reddit RSS)
- Uses your AI API key to generate a clean article page
- Publishes the output to GitHub Pages (**main / (root)**)

## Quick start (GitHub only)
1) Upload all files from this ZIP to your repository **root**
2) Edit `content_generator_pipeline/config.json`
3) Set GitHub Pages to **main / (root)**
4) Add the workflow file below and run it once
5) Open your Pages URL

## Step 1 — Upload files
On GitHub: **Add file → Upload files** → drop everything from the unzipped folder → **Commit changes**

## Step 2 — Configure API + feeds
Open: `content_generator_pipeline/config.json`

Edit these fields:
- `DEEPSEEK_API_KEY (GitHub secret)` → your AI API key (required)
- `site.base_url` → your GitHub Pages URL (recommended)
- (optional) `feeds.reddit_rss` → your RSS URL list

Commit changes.

## Step 3 — GitHub Pages setup
1) Repo → **Settings → Pages**
2) Source: **Deploy from a branch**
3) Branch: **main**
4) Folder: **/(root)**
5) Save
6) Copy the Pages URL shown there

## GitHub Actions setup

File name (create this file on GitHub):
`.github/workflows/run.yml`

How to add (GitHub UI):
1) Repo → **Add file → Create new file**
2) File name: `.github/workflows/run.yml`
3) Paste the YAML below
4) **Commit changes**

Workflow YAML (copy-paste):

```yaml
name: Run Content Generator (Lite)

on:
  workflow_dispatch:

permissions:
  contents: write

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt || true

      - name: Run generator (1 article)
        env:
          # Put your API key into config.json (recommended).
          # If your generator reads env, you can also pass it here.
          DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
        run: |
          python -m content_generator_pipeline.generate

      - name: Commit & push output
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

          # Only stage what the generator outputs (site files)
          git add index.html feed.xml sitemap.xml robots.txt assets articles og data/last_run.json data/articles.json processed_urls.txt || true

          git diff --cached --quiet && echo "No changes" && exit 0
          git commit -m "Update site output [skip ci]"
          git push
```

## Add secrets (optional)
Add this secret (required):
- Repo → Settings → Secrets and variables → Actions → New repository secret
- Name: `DEEPSEEK_API_KEY`
- Value: your DeepSeek API key

## How to run
1) Repo → **Actions**
2) **Run Content Generator (Lite)**
3) **Run workflow**

## Where to see the site (URL)
Repo → **Settings → Pages** → open the URL.

## Troubleshooting
No “Run workflow” button:
- The workflow file is not on the default branch
- YAML indentation is broken
- File path is not exactly `.github/workflows/run.yml`

Pages shows 404 / old content:
- Confirm Pages is **main / (root)**
- Wait 1–3 minutes
- Re-run the workflow once
