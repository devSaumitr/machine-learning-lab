# Web Scraping into a DataFrame — Study Notes (Day 18)

> Reference: `day18-pandas-dataframe-using-web-scraping` in the CampusX repo. Your work: `DAY18_Web-Scraping-Demo.ipynb` (AmbitionBox company list, page 4: names, ratings, counts).

## 1. Basic Idea

When no API or dataset exists, the data still sits inside web pages as HTML. Web scraping fetches the page and pulls the needed text out of its tags — Day 18 extracts company names, ratings, and review counts into Python lists (the future DataFrame columns).

## 2. Intuition and Mental Model

A web page is a labeled warehouse: every piece of text sits in a box (tag) with a label (class). Scraping is walking the aisles with a shopping list: all `h2` boxes for names, rating `div`s for scores, count `span`s for volumes.

## 3. Why Do We Need It?

Competitor lists, reviews, and niche catalogs often have no download button. Scraping is the last-resort collection method — slower and more fragile than APIs, but sometimes the only option.

## 4. Detailed Explanation

- **Look human first.** Sites block plain scripts, so send browser headers: `requests.get(url, headers={"User-Agent": "Mozilla/5.0 ..."})`. The `.text` attribute gives raw HTML.
- **Parse into a tree.** `BeautifulSoup(html, "lxml")` builds a searchable structure (needs the `lxml` package — install it in the environment, not inside the notebook).
- **Select, then read.** `soup.find_all("h2")` grabs every name tag; `card.find(...).get_text()` reads one field from one company card (`companyCardWrapper` divs). Narrow page → card → field, in that order.
- **Guard everything.** `find` returns `None` when a field is absent, and `None.get_text()` crashes. The one-line guard (`el.get_text() if el else None`) separates demos from real scrapers.

## 5. Examples

```python
import requests
from bs4 import BeautifulSoup
html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
soup = BeautifulSoup(html, "lxml")
names = [h.get_text(strip=True) for h in soup.find_all("h2")]
# per-card pattern:
# card = one companyCardWrapper div
# rating = card.find(...).get_text() if card.find(...) else None
```

## 6. Important Formulas

None — the pattern is headers → GET → parse → find → text.

## 7. Code Connection

`requests` fetches (same as Day 17, but `.text` not `.json()`); `BeautifulSoup` + `find_all`/`find`/`get_text` replace the JSON key lookup. Output lists are meant to become `pd.DataFrame({"company": ..., "rating": ...})` — the export your notebook has not done yet.

## 8. Interpretation

Scraped order is page order, not ranked truth. Counts like `(1.2L)` are display strings — strip text and convert before any math. One page proves the method, not the dataset.

## 9. Common Mistakes

1. No `User-Agent` → blocked or captcha'd.
2. `None.get_text()` crashes — always guard `find` results.
3. Brittle class names — sites rename them; re-verify selectors.
4. No delays/retries, hammering the server; ignoring robots.txt/terms of service.
5. Stopping at lists instead of building and saving the DataFrame (your notebook's current gap).

## 10. Machine Learning Connection

Scraped tables join the standard flow: DataFrame → clean → EDA → features. Because scraping is brittle, always snapshot results to CSV immediately so modeling never depends on the live site.

## 11. Interview and Conceptual Questions

1. Why is scraping the last resort? — Fragile, slow, legally/contractually sensitive; APIs and dumps are stable contracts.
2. What does the `User-Agent` header do? — Makes the request look like a browser to naive bot filters.
3. `find` vs `find_all`? — First match (possibly `None`) vs all matches (possibly empty list).
4. Why snapshot to CSV right away? — Decouples modeling from site changes and outages.

## 12. Quick Revision

Headers → GET → `BeautifulSoup(lxml)` → page → card → field → guard `None` → DataFrame → CSV. Verify selectors, respect the site, save early.
