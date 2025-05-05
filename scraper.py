import requests
import json
import time
import pandas as pd
import urllib.parse
import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import nest_asyncio

nest_asyncio.apply()

# 🛠️ API Settings
API_TOKEN = 'apify_api_BfGpFJxILZZR2KXrLwQK5jb4psiPuP1P7hW3'  # change this
POST_SCRAPER_ID = 'apify~facebook-posts-scraper'
COMMENT_SCRAPER_ID = 'apify~facebook-comments-scraper'


# ================== Extract Article Details ==================
def extract_article_details(article_url):
    """Extracts title, description, image, long description, and category from an article page."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(article_url, headers=headers, timeout=100)

        if response.status_code != 200:
            return "Unavailable", "Unavailable", "Unavailable", "Unavailable", "Unavailable"

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title
        title = soup.title.text.strip() if soup.title else "Unavailable"

        # Extract short description (meta description)
        description_tag = soup.find("meta", attrs={"name": "description"})
        short_description = description_tag["content"].strip() if description_tag else "Unavailable"

        # Extract article image (og:image)
        image_tag = soup.find("meta", attrs={"property": "og:image"})
        article_image = image_tag["content"].strip() if image_tag else "Unavailable"

        # Extract long description (skip the advertisement paragraph)
        long_description = ""
        paragraphs = soup.select('div[dir="rtl"].bng-link-body p')

        for p in paragraphs:
            if "لمشاهدة جميع المباريات والأحداث الرياضية" in p.text:
                continue
            long_description += p.text.strip() + "\n\n"

        long_description = long_description.strip() if long_description else "Unavailable"

        # Extract category from breadcrumb (in original Arabic)
        category = "Unavailable"
        breadcrumb = soup.find('ul', class_='md:bng-flex')
        if breadcrumb:
            # Get the first sports category after الرئيسية (Home)
            for item in breadcrumb.find_all('li')[1:]:  # Skip first item (الرئيسية)
                link = item.find('a')
                if link:
                    category = link.get_text(strip=True)
                    break  # Take the first category after الرئيسية

        return title, short_description, article_image, long_description, category

    except Exception as e:
        print(f"❌ Error extracting data from {article_url}: {e}")
        return "Unavailable", "Unavailable", "Unavailable", "Unavailable", "Unavailable"


# ================== Dailymotion Video Extraction ==================
async def get_dailymotion_video_url(article_url):
    """Extracts embedded Dailymotion video URLs from an article."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        print(f"🔍 Loading article: {article_url}")

        await page.goto(article_url, timeout=90000)
        await page.wait_for_timeout(3000)

        await page.screenshot(path="screenshot.png")
        print("📸 Screenshot saved: 'screenshot.png'")

        video_elements = await page.query_selector_all("iframe")
        video_urls = [await video.get_attribute("src") for video in video_elements if
                      "dailymotion" in (await video.get_attribute("src") or "")]

        await browser.close()
        return video_urls[0] if video_urls else "No video"


# ================== Facebook Image URL Cleaning ==================
def clean_facebook_image_url(image_url):
    """Cleans Facebook image URLs to get the original source."""
    if not image_url or "flim" in str(image_url):
        return None

    if isinstance(image_url, list):
        image_url = image_url[0] if image_url else None

    if isinstance(image_url, dict):
        image_url = image_url.get("thumbnail")

    if not isinstance(image_url, str):
        return None

    if "url=" in image_url:
        parsed_url = urllib.parse.parse_qs(urllib.parse.urlparse(image_url).query)
        original_url = parsed_url.get("url", [None])[0]
        return original_url.split("?")[0] if original_url else None

    return image_url.split("?")[0]


# ================== Facebook Post Scraping ==================
def fetch_facebook_posts():
    """Runs Apify scraper to fetch Facebook posts."""
    all_posts = []

    post_api_url = f"https://api.apify.com/v2/acts/{POST_SCRAPER_ID}/runs?token={API_TOKEN}"
    post_payload = {
        "maxPostCount": 10,
        "resultsLimit": 10,
        "startUrls": [{"url": "https://www.facebook.com/beinsports/posts"}],
        "proxyConfiguration": {"useApifyProxy": True},
        "scrapePostsFromFeed": True,
        "scrapeDateThresholdHours": 720,
        "scrollTimeoutSecs": 60,
        "scrollCount": 10
    }

    post_response = requests.post(post_api_url, json=post_payload)
    if post_response.status_code != 201:
        print("❌ Error starting post extraction.")
        return []

    run_id = post_response.json().get('data', {}).get('id')
    if not run_id:
        print("❌ Failed to retrieve run ID.")
        return []

    post_status_url = f"https://api.apify.com/v2/acts/{POST_SCRAPER_ID}/runs/{run_id}"

    while True:
        run_status = requests.get(f"{post_status_url}?token={API_TOKEN}").json()
        status = run_status.get('data', {}).get('status')

        if status == 'SUCCEEDED':
            print("✅ Post extraction completed.")
            break
        elif status in ['FAILED', 'ABORTED', 'TIMED-OUT']:
            print("❌ Post extraction failed.")
            return []

        print("⏳ Processing...")
        time.sleep(15)

    dataset_id = run_status.get('data', {}).get('defaultDatasetId')
    if not dataset_id:
        print("❌ Failed to retrieve dataset ID.")
        return []

    dataset_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?format=json&clean=1&token={API_TOKEN}"
    posts = requests.get(dataset_url).json()

    return posts or []


# ================== Facebook Comments Scraping ==================
def fetch_facebook_comments(post_url, post_id):
    print(f"🔍 Fetching comments for post {post_id}...")  # Log pour voir quel post est traité
    comment_api_url = f"https://api.apify.com/v2/acts/{COMMENT_SCRAPER_ID}/runs?token={API_TOKEN}"
    comment_payload = {
        "startUrls": [{"url": post_url}],
        "proxyConfiguration": {"useApifyProxy": True},
        "maxComments": 2000,
        "sortBy": "top"
    }

    comment_response = requests.post(comment_api_url, json=comment_payload)
    if comment_response.status_code != 201:
        print(f"❌ Error launching comment scraper for post {post_id}")
        return []

    comment_run_id = comment_response.json().get("data", {}).get("id")
    if not comment_run_id:
        print(f"❌ Failed to get run ID for comments of post {post_id}")
        return []

    comment_status_url = f"https://api.apify.com/v2/acts/{COMMENT_SCRAPER_ID}/runs/{comment_run_id}"

    while True:
        comment_run_status = requests.get(f"{comment_status_url}?token={API_TOKEN}").json()
        status = comment_run_status.get("data", {}).get("status")

        if status == "SUCCEEDED":
            break
        elif status in ["FAILED", "ABORTED"]:
            print(f"❌ Comment scraper failed for post {post_id}")
            return []

        print(f"⏳ Waiting for comments of post {post_id}...")
        time.sleep(15)

    comment_dataset_id = comment_run_status.get("data", {}).get("defaultDatasetId")
    if not comment_dataset_id:
        print(f"❌ Failed to get dataset ID for comments of post {post_id}")
        return []

    comment_dataset_url = f"https://api.apify.com/v2/datasets/{comment_dataset_id}/items?format=json&clean=1&token={API_TOKEN}"
    return requests.get(comment_dataset_url).json()


# 🔹 Exécuter le script principal
def run_script():
    posts = fetch_facebook_posts()
    if not posts:
        print("❌ No posts found.")
        exit()

    print(f"✅ Extracted {len(posts)} Facebook posts.")

    # ✅ Initialiser les listes ici
    post_data = []
    comment_data = []

    for post in posts:
        try:
            post_id = str(post.get('postId', ''))
            post_title = post.get('title') or post.get('text') or "No content"
            post_title = (post_title[:100] + '...') if len(post_title) > 100 else post_title
            article_link = post.get('link')

            image_url = clean_facebook_image_url(post.get('media') or post.get('picture'))
            article_title, short_description, article_image, long_description, category = extract_article_details(
                article_link) if article_link else ("Unavailable", "Unavailable", None, "Unavailable", "Unavailable")
            video_link = asyncio.run(get_dailymotion_video_url(article_link)) if article_link else "No video"

            # Skip if no category or no media content
            if category == "Unavailable" or not any([image_url, article_image, video_link != "No video"]):
                continue

            post_data.append({
                'post_id': post_id,
                'title': post_title,
                'article_link': article_link,
                'article_title': article_title,
                'short_description': short_description,
                'long_description': long_description,
                'category': category,  # This will always have a valid category now
                'image': image_url or article_image,
                'video_link': video_link,
                'likes_count': post.get('likes', 0),
                'comments_count': post.get('comments', 0),
                'shares_count': post.get('shares', 0)  # Added shares count here
            })

            # Only fetch comments for posts with valid categories
            if article_link and category != "Unavailable":
                comments = fetch_facebook_comments(post.get("url", f"https://www.facebook.com/{post_id}"), post_id)
                for comment in comments:
                    comment_data.append({
                        'post_id': post_id,
                        'comment_id': comment.get('id', ''),
                        'author': comment.get('author', 'Unknown'),
                        'text': comment.get('text', 'No content'),
                        'likes': comment.get('likes', 0),
                        'timestamp': comment.get('timestamp', '')
                    })

        except Exception as e:
            print(f"❌ Error processing post {post.get('postId')}: {e}")

    # ✅ Sauvegarde des données
    pd.DataFrame(post_data, dtype=str).to_csv("facebook_posts.csv", index=False)
    pd.DataFrame(comment_data, dtype=str).to_csv("facebook_comments.csv", index=False)

    print("✅ Posts and comments saved!")


# 🕑 Loop to run the script every hour
while True:
    run_script()
    print("⏳ Waiting for the next run...")
    time.sleep(3600)  # Sleep for 1 hour