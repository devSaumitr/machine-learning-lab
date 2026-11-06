# Day 18 - Fetching Data with Web Scraping

> CampusX 100 Days of Machine Learning, Day 18.
> One unified chapter covering the official AmbitionBox notebook and the separate personal notebook.

### How to read this chapter

**Part A** teaches the problem, HTML foundations, the `requests` + BeautifulSoup workflow, the official 1,000-page experiment, the local notebook, data cleaning, ethics, and essential interview material. It is sufficient to understand and implement the day without reading Part B.

**Part B** is optional advanced reading: HTML/CSS reference, BeautifulSoup internals, table-extraction alternatives, robust scraping patterns, a cell-level source-fidelity audit, extended comparisons, an extended interview bank, and extra examination practice.

The day has no ML model and no train/test split. Its quantitative foundations are **cards per page, pages, rows, columns, response sizes, and the cost of repeated concatenation**.

---

# Part A - Main Learning Chapter

## 1. The problem: the data is inside HTML, not a file

Day 17 fetched JSON from an API. Day 18 handles the harder case: a website displays data for humans, but exposes no clean download or API.

The page is HTML. The data is wrapped in tags, classes, and layout elements. To analyse it, you must:

1. fetch the HTML;
2. parse it into a tree;
3. find the repeated card or row that represents one record;
4. extract the fields from each card;
5. build a DataFrame;
6. clean the strings into usable types;
7. save a snapshot.

```text
URL
 |
 v
requests.get(url, headers=..., timeout=...)
 |
 v
HTML text
 |
 v
BeautifulSoup(html, "lxml")
 |
 v
page -> card -> field
 |
 v
Python lists / dictionaries
 |
 v
pandas DataFrame
 |
 v
clean -> snapshot
```

The official notebook scrapes AmbitionBox company cards. The local notebook demonstrates the same pattern against the current AmbitionBox markup.

### 1.1 Why scraping is a last resort

Scraping is:

- **brittle:** class names and page layouts change;
- **slow:** each page is a separate request;
- **legally sensitive:** terms of service, robots.txt, copyright, and personal data may apply;
- **operationally fragile:** sites block, rate-limit, or show verification pages;
- **hard to reproduce:** live pages change between runs.

Use an API or an official dataset when one exists. Use scraping when the page is the only source and you have the right to collect the data.

### 1.2 What the official notebook actually demonstrates

The official notebook:

- requests `https://www.ambitionbox.com/list-of-companies?page=1`;
- parses the HTML with `BeautifulSoup(webpage, "lxml")`;
- counts 30 company cards, 30 ratings, and 30 review counts on the first page;
- builds a 30-row, 7-column DataFrame;
- loops `range(1, 1001)`, i.e. 1,000 pages;
- appends each page to `final`;
- reports `final.shape == (30000, 7)`.

It does **not** save a CSV. It does not check status codes, set a timeout, sleep between requests, or guard every field cleanly.

### 1.3 What the local notebook demonstrates

The local `my_notebook.ipynb`:

- imports `requests`, `BeautifulSoup`, and `pandas`;
- sends a Chrome `User-Agent`;
- requests a page with a `campaign=...` URL;
- searches for `companyCardWrapper` cards, `rating_text` ratings, and `companyCardWrapper__companyRatingCount` counts;
- prints fields from each card;
- has no saved outputs and no DataFrame/CSV export.

It is a selector demo, not a complete scraper.

---

## 2. Foundations: HTML, the DOM, tags, and classes

### 2.1 HTML is a tree

HTML is text, but a parser turns it into a tree of elements:

```html
<html>
  <body>
    <div class="companyCardWrapper">
      <h2 class="companyCardWrapper__companyName">TCS</h2>
      <div class="rating_text">3.9</div>
      <span class="companyCardWrapper__companyRatingCount">(17.1k)</span>
    </div>
  </body>
</html>
```

The tree is:

```text
html
└── body
    └── div.companyCardWrapper
        ├── h2.companyCardWrapper__companyName  -> "TCS"
        ├── div.rating_text                    -> "3.9"
        └── span.companyCardWrapper__companyRatingCount -> "(17.1k)"
```

Each element has:

- a **tag name** (`div`, `h2`, `span`);
- **attributes** (`class="..."`, `id="..."`, `href="..."`);
- **text content**;
- **children**.

### 2.2 Tags

Common tags:

| Tag | Typical use |
|---|---|
| `div` | generic container |
| `span` | inline text container |
| `h1`, `h2`, `h3` | headings |
| `p` | paragraph |
| `a` | link |
| `table`, `tr`, `td`, `th` | tabular data |
| `ul`, `li` | lists |

The official notebook uses `div`, `h2`, `p`, and `a`. The local notebook uses `div`, `h2`, and `span`.

### 2.3 Classes

A class is an attribute used to group elements for styling and scripting:

```html
<div class="companyCardWrapper">
```

In CSS, `.companyCardWrapper` selects elements with that class. In BeautifulSoup:

```python
soup.find_all("div", class_="companyCardWrapper")
```

`class_` has an underscore because `class` is a Python keyword.

An element can have multiple classes:

```html
<div class="companyCardWrapper companyCardWrapper--premium">
```

`class_="companyCardWrapper"` matches the element because it has that class among its classes. For exact multi-class matching or complex combinations, use `soup.select(...)`.

### 2.4 IDs and attributes

An `id` should be unique on a page:

```html
<div id="company-123">
```

Access attributes with:

```python
tag.get("href")
tag["href"]
tag.get("class")
```

Use `.get()` when the attribute may be absent; `tag["href"]` raises `KeyError` if missing.

### 2.5 CSS selectors

BeautifulSoup supports CSS selectors through `select` and `select_one`:

```python
soup.select("div.companyCardWrapper")
soup.select_one("div.companyCardWrapper h2")
soup.select("a[href]")
```

CSS selectors are compact and expressive, but they are just as brittle as `find_all` when class names change. Choose one style and be consistent.

### 2.6 The DOM is not the HTML string

The HTML string is the raw text returned by `requests`. The DOM/tree is the parsed object built by BeautifulSoup. Methods such as `find_all` exist on the parsed tree, not on the string.

```python
webpage = response.text          # str
soup = BeautifulSoup(webpage, "lxml")  # BeautifulSoup object
```

This distinction matters when debugging: `"companyCardWrapper" in webpage` checks the raw string; `soup.find(...)` searches the parsed tree.

---

## 3. Fetching the page with `requests`

### 3.1 The basic request

```python
import requests

url = "https://www.ambitionbox.com/list-of-companies?page=1"
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    )
}

response = requests.get(url, headers=headers, timeout=30)
response.raise_for_status()
webpage = response.text
```

The `User-Agent` header identifies the client. Some sites return a 403 or a verification page to clients that look like scripts. A browser-like user agent can avoid naive filters, but it is not a right to bypass access controls.

### 3.2 Status codes are not enough

A 200 response can still be a verification page, a cookie wall, or an error page styled as HTML. The official notebook does not check status at all. The local notebook also does not check.

A robust scraper checks:

1. the HTTP status;
2. whether the expected marker exists in the HTML;
3. whether the expected number of cards was found;
4. whether the extracted fields are non-empty.

```python
if response.status_code != 200:
    raise RuntimeError(f"HTTP {response.status_code}")

if "companyCardWrapper" not in response.text:
    raise RuntimeError("Expected company cards not found; page may be blocked or changed")
```

### 3.3 Timeouts

```python
response = requests.get(url, headers=headers, timeout=30)
```

Without a timeout, a slow server can hang the notebook. A timeout is not a retry policy; it only bounds the wait.

### 3.4 Headers

Common headers:

| Header | Purpose |
|---|---|
| `User-Agent` | identify the client |
| `Accept` | requested content type |
| `Accept-Language` | language preference |
| `Referer` | previous page |
| `Cookie` | session state |
| `Authorization` | token-based access |

Do not copy cookies or tokens from a browser session unless you are authorized to do so. Do not commit them.

### 3.5 Why the official markdown mentions 403

The official notebook has a markdown cell:

```text
# if response code is 403
- headers={'User-Agent': 'Mozilla/5.0 ...'}
- requests.get('url', headers=headers).text
```

This is a common workaround for naive bot blocking. It is not a guarantee. A site may still:

- return a CAPTCHA;
- return a verification page with status 200;
- require JavaScript;
- block the IP;
- change its HTML.

A 2026-11-04 supplementary check of the AmbitionBox list page returned HTTP 200 with the title `Verify - AmbitionBox` and none of the target classes. That is a verification page, not the company list. It is a concrete example of why status 200 alone is insufficient.

---

## 4. Parsing with BeautifulSoup

### 4.1 Create the soup

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(webpage, "lxml")
```

The second argument is the parser:

| Parser | Package | Characteristics |
|---|---|---|
| `"html.parser"` | standard library | no extra install; moderate speed |
| `"lxml"` | `lxml` | fast; lenient; common choice |
| `"html5lib"` | `html5lib` | very lenient; slower; browser-like parsing |

Install `lxml` in the environment:

```bash
pip install lxml
```

The local notebook contains a code cell `pip install lxml` without a leading `!`. That is not valid Python. Install it in the terminal or use `%pip install lxml` in Jupyter.

### 4.2 `find` and `find_all`

```python
first_h2 = soup.find("h2")              # Tag or None
all_h2 = soup.find_all("h2")            # ResultSet (list-like)
```

- `find` returns the first match or `None`.
- `find_all` returns all matches or an empty list-like object.

The most common bug is calling a method on `None`:

```python
name = card.find("h2").get_text(strip=True)
```

If the card has no `h2`, `card.find("h2")` is `None`, and `.get_text()` raises `AttributeError`.

Guard it:

```python
name_el = card.find("h2", class_="companyCardWrapper__companyName")
name = name_el.get_text(strip=True) if name_el else None
```

### 4.3 Filtering by class

```python
soup.find_all("p", class_="rating")
soup.find_all("a", class_="review-count")
soup.find_all("div", class_="company-content-wrapper")
```

The official notebook uses these three selectors. On the saved first page:

- 30 `p.rating` elements;
- 30 `a.review-count` elements;
- 30 `div.company-content-wrapper` elements.

The counts agreeing is a useful sanity check: one rating and one review count per company card.

### 4.4 Reading text

```python
tag.get_text()
tag.get_text(strip=True)
tag.text
```

- `get_text()` returns all descendant text.
- `get_text(strip=True)` strips surrounding whitespace from each text fragment and joins.
- `.text` is a property that returns the text; it is convenient but less explicit.

For scraped fields, `get_text(strip=True)` is usually the safest default. It removes the newlines and indentation that make scraped values messy.

### 4.5 Reading attributes

```python
link = tag.get("href")
classes = tag.get("class")       # list of class names
identifier = tag.get("id")
```

`tag["href"]` works when the attribute exists; `.get()` is safer when it may not.

### 4.6 A static example

This example uses a local HTML string, so it is reproducible without a network:

```python
from bs4 import BeautifulSoup

html = """
<div class="companyCardWrapper">
  <h2 class="companyCardWrapper__companyName">TCS</h2>
  <div class="rating_text">3.9</div>
  <span class="companyCardWrapper__companyRatingCount">(17.1k)</span>
</div>
"""

soup = BeautifulSoup(html, "html.parser")
card = soup.find("div", class_="companyCardWrapper")

name = card.find("h2", class_="companyCardWrapper__companyName")
rating = card.find("div", class_="rating_text")
count = card.find("span", class_="companyCardWrapper__companyRatingCount")

print(name.get_text(strip=True))
print(rating.get_text(strip=True))
print(count.get_text(strip=True))
```

Output:

```text
TCS
3.9
(17.1k)
```

The flow is page -> card -> field. Do not search the whole page for every field if the page has repeated cards; narrow to one card first.

### 4.7 Page-level versus card-level extraction

Page-level:

```python
names = [h.get_text(strip=True) for h in soup.find_all("h2")]
```

This works when every `h2` is a company name. It fails when the page has other `h2` headings.

Card-level:

```python
cards = soup.find_all("div", class_="companyCardWrapper")
rows = []

for card in cards:
    name_el = card.find("h2", class_="companyCardWrapper__companyName")
    rating_el = card.find("div", class_="rating_text")
    rows.append({
        "name": name_el.get_text(strip=True) if name_el else None,
        "rating": rating_el.get_text(strip=True) if rating_el else None,
    })
```

Card-level extraction keeps fields aligned. If one card is missing a rating, the name and rating still belong to the same row. Page-level lists can silently misalign when one field is missing.

---

## 5. The official notebook: page 1 and the 1,000-page loop

### 5.1 Fetch and parse

```python
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np

webpage = requests.get(
    "https://www.ambitionbox.com/list-of-companies?page=1"
).text
soup = BeautifulSoup(webpage, "lxml")
```

The official code has no headers, timeout, or status check on this first request. The markdown cell above it suggests a `User-Agent` workaround for 403, but the code does not apply it.

### 5.2 The first heading

```python
soup.find_all("h1")[0].text
```

Saved output:

```text
'List of companies in India'
```

This is a fragile pattern. If the page has no `h1`, `[0]` raises `IndexError`. Prefer:

```python
h1 = soup.find("h1")
title = h1.get_text(strip=True) if h1 else None
```

### 5.3 Names from `h2`

```python
for i in soup.find_all("h2"):
    print(i.text.strip())
```

The saved output lists 30 companies, beginning with:

```text
TCS
Accenture
ICICI Bank
Cognizant
HDFC Bank
Capgemini
Tech Mahindra
HCL Technologies
L&T
Infosys
```

The full saved list contains 30 names. The count is not printed, but the DataFrame shape later confirms 30 rows.

### 5.4 Why `find_all("p")` is too broad

```python
for i in soup.find_all("p"):
    print(i.text.strip())
```

The saved output is a long stream containing:

- the page summary `477,887 unique companies found`;
- sort labels;
- every company rating;
- company type, headquarters, age, and employee count;
- long company descriptions;
- footer text.

It is not a table. It mixes fields and records. This is why the notebook later narrows to cards.

### 5.5 Counting the target elements

```python
len(soup.find_all("p", class_="rating"))
len(soup.find_all("a", class_="review-count"))
```

Saved outputs:

```text
30
30
```

The two counts agree. The notebook then finds the card container:

```python
company = soup.find_all("div", class_="company-content-wrapper")
len(company)
```

Saved output:

```text
30
```

Three independent counts of 30 are a useful consistency check. If one count differs, the page structure or selector assumptions are wrong.

### 5.6 Building the first-page DataFrame

```python
name = []
rating = []
reviews = []
ctype = []
hq = []
how_old = []
no_of_employee = []

for i in company:
    name.append(i.find("h2").text.strip())
    rating.append(i.find("p", class_="rating").text.strip())
    reviews.append(i.find("a", class_="review-count").text.strip())
    ctype.append(i.find_all("p", class_="infoEntity")[0].text.strip())
    hq.append(i.find_all("p", class_="infoEntity")[1].text.strip())
    how_old.append(i.find_all("p", class_="infoEntity")[2].text.strip())
    no_of_employee.append(i.find_all("p", class_="infoEntity")[3].text.strip())

df = pd.DataFrame({
    "name": name,
    "rating": rating,
    "reviews": reviews,
    "company_type": ctype,
    "Head_Quarters": hq,
    "Company_Age": how_old,
    "No_of_Employee": no_of_employee,
})
```

Saved shape:

```text
(30, 7)
```

The seven columns are:

| Column | Example | Type as scraped |
|---|---|---|
| `name` | TCS | string |
| `rating` | 3.9 | string |
| `reviews` | 17.1k Reviews | string |
| `company_type` | Public | string |
| `Head_Quarters` | Mumbai + 164 more | string |
| `Company_Age` | 52 years old | string |
| `No_of_Employee` | 10000+ employees | string |

Every value is a string, including the numbers. Scraped data always needs type conversion before analysis.

### 5.7 The saved first-page values

A few saved rows:

| name | rating | reviews | company_type | Head_Quarters | Company_Age | No_of_Employee |
|---|---:|---|---|---|---|---|
| TCS | 3.9 | 17.1k Reviews | Public | Mumbai + 164 more | 52 years old | 10000+ employees |
| Accenture | 4.0 | 15k Reviews | Private | Dublin + 98 more | 31 years old | 10000+ employees |
| ICICI Bank | 4.1 | 13.8k Reviews | Public | Mumbai + 743 more | 26 years old | 10000+ employees |
| Cognizant | 3.9 | 12.9k Reviews | Private | Teaneck + 46 more | 26 years old | 10000+ employees |
| HDFC Bank | 4.0 | 12k Reviews | Public | Mumbai + 729 more | 26 years old | 10000+ employees |

These are saved outputs from the official notebook, not live values.

### 5.8 The 1,000-page loop

```python
final = pd.DataFrame()

for j in range(1, 1001):
    webpage = requests.get(
        "https://www.ambitionbox.com/list-of-companies?page={}".format(j)
    ).text
    soup = BeautifulSoup(webpage, "lxml")
    company = soup.find_all("div", class_="company-content-wrapper")

    name = []
    rating = []
    reviews = []
    ctype = []
    hq = []
    how_old = []
    no_of_employee = []

    for i in company:
        try:
            name.append(i.find("h2").text.strip())
        except:
            name.append(np.nan)

        try:
            rating.append(i.find("p", class_="rating").text.strip())
        except:
            rating.append(np.nan)

        try:
            reviews.append(i.find("a", class_="review-count").text.strip())
        except:
            reviews.append(np.nan)

        try:
            ctype.append(i.find_all("p", class_="infoEntity")[0].text.strip())
        except:
            ctype.append(np.nan)

        try:
            hq.append(i.find_all("p", class_="infoEntity")[1].text.strip())
        except:
            hq.append(np.nan)

        try:
            how_old.append(i.find_all("p", class_="infoEntity")[2].text.strip())
        except:
            how_old.append(np.nan)

        try:
            no_of_employee.append(i.find_all("p", class_="infoEntity")[3].text.strip())
        except:
            no_of_employee.append(np.nan)

    df = pd.DataFrame({
        "name": name,
        "rating": rating,
        "reviews": reviews,
        "company_type": ctype,
        "Head_Quarters": hq,
        "Company_Age": how_old,
        "No_of_Employee": no_of_employee,
    })

    final = final.append(df, ignore_index=True)
```

### 5.9 The row arithmetic

`range(1, 1001)` produces page numbers 1 through 1,000:

$$
1001-1=1{,}000.
$$

The first page has 30 company cards. If every page contributes 30 rows:

$$
1{,}000 \times 30 = 30{,}000.
$$

The saved output is:

```text
final.shape
(30000, 7)
```

This matches exactly. The average rows per page is:

$$
\frac{30{,}000}{1{,}000}=30.
$$

The page also displays:

```text
477,887 unique companies found
```

If that count and the 30-per-page layout are both accurate, the number of pages needed to cover the directory is:

$$
\frac{477{,}887}{30}\approx 15{,}929.57.
$$

The official loop collects 1,000 pages, i.e. 30,000 rows, which is:

$$
\frac{30{,}000}{477{,}887}\approx 0.06278
$$

or about **6.28%** of the displayed company count. The official dataset is therefore a subset, not the whole directory. This is an inference from the displayed count and page size, not a guarantee that every page has exactly 30 cards.

### 5.10 The `df` versus `final` distinction

Inside the loop, `df` is reassigned on every page. After the loop:

- `df` is the **last page** (30 rows);
- `final` is **all pages** (30,000 rows).

The saved `df.sample(5)` shows companies such as Bank of America, R.R. Donnelley, Mahindra Holidays, GVK Emergency Management, and Itc Infotech. Those are not the first-page companies. This confirms that `df` is the last page, not the full dataset.

The sample is random and no `random_state` is set, so the five rows can change between runs.

### 5.11 The `append` problem

`final.append(df, ignore_index=True)` is removed in pandas 2.0. The official notebook was executed with an older pandas. In pandas 3.0.5, it raises `AttributeError`.

The modern pattern is to collect frames and concatenate once:

```python
frames = []

for page in range(1, 1001):
    response = requests.get(url, params={"page": page}, headers=headers, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    cards = soup.find_all("div", class_="companyCardWrapper")

    rows = []
    for card in cards:
        name_el = card.find("h2", class_="companyCardWrapper__companyName")
        rating_el = card.find("div", class_="rating_text")
        rows.append({
            "name": name_el.get_text(strip=True) if name_el else None,
            "rating": rating_el.get_text(strip=True) if rating_el else None,
        })

    frames.append(pd.DataFrame(rows))
    time.sleep(1)

final = pd.concat(frames, ignore_index=True)
```

### 5.12 Why repeated `append` is expensive

Each `append` creates a new DataFrame and copies the accumulated data. If page $k$ adds 30 rows, the total number of row copies is roughly:

$$
30 + 60 + \cdots + 30{,}000
= 30 \times \frac{1{,}000 \times 1{,}001}{2}
= 15{,}015{,}000.
$$

That is about 15 million row copies for a final frame of 30,000 rows. Collecting frames in a list and concatenating once avoids most of that copying. This is the algorithmic reason `append` was removed.

### 5.13 The bare `except`

The loop uses:

```python
try:
    ...
except:
    ...
```

A bare `except:` catches `BaseException`, including `KeyboardInterrupt` and `SystemExit`. It also hides the actual error and makes debugging hard. Prefer:

```python
except (AttributeError, IndexError, TypeError) as exc:
    ...
```

and log the page and card index.

### 5.14 Positional `infoEntity` indexing

```python
i.find_all("p", class_="infoEntity")[0]  # company type
i.find_all("p", class_="infoEntity")[1]  # headquarters
i.find_all("p", class_="infoEntity")[2]  # age
i.find_all("p", class_="infoEntity")[3]  # employees
```

This assumes every card has exactly four `infoEntity` paragraphs in the same order. If one is missing, the `try/except` appends `NaN` for that field, but later indices may refer to shifted positions. The result can be silently wrong rather than missing.

A more robust approach is to find fields by their label or a dedicated class, not by position. If the site provides no labels, validate the length before indexing:

```python
entities = i.find_all("p", class_="infoEntity")
ctype = entities[0].get_text(strip=True) if len(entities) > 0 else None
hq = entities[1].get_text(strip=True) if len(entities) > 1 else None
age = entities[2].get_text(strip=True) if len(entities) > 2 else None
employees = entities[3].get_text(strip=True) if len(entities) > 3 else None
```

### 5.15 No snapshot

The official notebook never writes `final` to disk. If the kernel dies, the 30,000 rows are lost. A scraper should save early and often:

```python
final.to_csv("datasets/ambitionbox_companies.csv", index=False)
```

Use `index=False` to avoid an `Unnamed: 0` column like the one in Day 17's `movies.csv`.

---

## 6. The local notebook: current selectors, no outputs

### 6.1 The fetch

```python
import pandas as pd
import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    )
}

url = "https://www.ambitionbox.com/list-of-companies?campaign=desktop_nav%3E%3Fpage%20%3D4"
webpage = requests.get(url, headers=headers).text
```

The URL is not a normal `?page=4` request. It encodes `?page =4` inside the `campaign` parameter:

```text
campaign=desktop_nav%3E%3Fpage%20%3D4
```

Decoded, that is roughly `campaign=desktop_nav>?page =4`. The site may ignore it or treat it as a campaign value. If the intent was page 4, use:

```python
params = {"page": 4}
response = requests.get(base_url, params=params, headers=headers, timeout=30)
```

### 6.2 The selectors

```python
company_cards = soup.find_all("div", class_="companyCardWrapper")
ratings = soup.find_all("div", class_="rating_text")
rating_count = soup.find_all(
    "span",
    class_="companyCardWrapper__companyRatingCount",
)
```

These are newer class names than the official notebook's:

| Official (older) | Local (newer) |
|---|---|
| `company-content-wrapper` | `companyCardWrapper` |
| `p.rating` | `div.rating_text` |
| `a.review-count` | `span.companyCardWrapper__companyRatingCount` |
| `p.infoEntity` | not used |

The change is exactly the fragility problem: a site redesign can invalidate every selector.

### 6.3 The per-card extraction

```python
for card in company_cards:
    company_name = card.find(
        "h2",
        class_="companyCardWrapper__companyName",
    )
    rating = card.find(
        "div",
        class_="rating_text",
    )
    rating_count = card.find(
        "span",
        class_="companyCardWrapper__companyRatingCount",
    )

    print(company_name.get_text(strip=True))
    print(rating.get_text(strip=True))
    print(rating_count.get_text(strip=True))
```

There is no `None` guard. If any card lacks one of the three elements, `.get_text()` raises `AttributeError` and the loop stops. The local notebook's own takeaway cell identifies this gap.

### 6.4 No saved outputs

The local notebook has no saved outputs for the fetch, the selectors, or the per-card loop. The only non-empty output is a small HTML example in markdown. Therefore:

- we cannot claim the selectors worked on the saved run;
- we cannot claim a row count from the local notebook;
- the local notebook is a code demonstration, not evidence of a successful scrape.

A 2026-11-04 supplementary live check returned a `Verify - AmbitionBox` page with none of the target classes. That is consistent with a bot-verification wall, not with a successful scrape.

### 6.5 The invalid install cell

The local notebook contains:

```python
pip install lxml
```

as a code cell. That is not valid Python. In Jupyter, use:

```python
%pip install lxml
```

or install from the terminal:

```bash
pip install lxml
```

### 6.6 What the local notebook does well

- It uses a browser-like `User-Agent`.
- It narrows to cards before reading fields.
- It uses `get_text(strip=True)`.
- It separates name, rating, and count.
- It documents the missing-field crash as an experiment.

### 6.7 What the local notebook does not do

- It does not check the status.
- It does not set a timeout.
- It does not validate the number of cards.
- It does not guard `None`.
- It does not paginate.
- It does not build a DataFrame.
- It does not save a CSV.
- It does not handle the verification page.
- It uses a malformed page URL.
- It has no saved outputs.

---

## 7. From scraped strings to usable features

### 7.1 Why conversion is necessary

Scraped values are strings:

```text
"3.9"
"17.1k Reviews"
"52 years old"
"10000+ employees"
"Mumbai + 164 more"
```

You cannot compute a mean rating, sort by review count, or plot age against rating until these become numbers. The DataFrame is only the first step.

### 7.2 Ratings

```python
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
```

`errors="coerce"` turns unparseable values into `NaN` instead of raising. Then validate:

```python
print(df["rating"].min(), df["rating"].max())
print(df["rating"].isna().sum())
```

The saved first page has ratings from 3.4 to 4.7. Ratings are on a 5-point scale in this snapshot.

### 7.3 Review counts

```python
import re

def parse_count(text):
    if text is None:
        return None
    text = text.replace("Reviews", "").strip()
    if text.endswith("k"):
        return float(text[:-1]) * 1_000
    if text.endswith("L"):
        return float(text[:-1]) * 100_000
    if text.endswith("Cr"):
        return float(text[:-2]) * 10_000_000
    return float(text.replace(",", ""))

df["reviews_numeric"] = df["reviews"].map(parse_count)
```

Examples:

| Text | Numeric |
|---|---:|
| `17.1k Reviews` | 17,100 |
| `15k Reviews` | 15,000 |
| `13.8k Reviews` | 13,800 |
| `1.2L Reviews` | 120,000 |
| `1.2Cr Reviews` | 12,000,000 |

The values are rounded display counts, not exact counts. Treat them as approximate.

### 7.4 Company age

```python
df["company_age_years"] = (
    df["Company_Age"]
    .str.extract(r"(\d+)")[0]
    .astype(float)
)
```

`"52 years old"` becomes `52.0`. If the text is missing, the result is `NaN`.

### 7.5 Employee count

```python
df["employees_lower_bound"] = (
    df["No_of_Employee"]
    .str.extract(r"(\d+)")[0]
    .astype(float)
)
```

`"10000+ employees"` becomes `10000.0`. This is a lower bound, not an exact count. If the site shows ranges such as `"501-1000 employees"`, a single regex is not enough; parse the range explicitly.

### 7.6 Headquarters

```python
df["hq_city"] = (
    df["Head_Quarters"]
    .str.split(" + ", n=1, regex=False)
    .str[0]
)
```

`"Mumbai + 164 more"` becomes `"Mumbai"`. The `+ 164 more` is a count of additional locations, not part of the city name.

### 7.7 Company type

```python
df["company_type"] = df["company_type"].astype("category")
```

Categories are useful for grouped analysis and later encoding. The saved first page includes `Public`, `Private`, and `Govt agency`.

### 7.8 Missing values

After conversion, inspect:

```python
df.isna().sum()
```

Missing values can come from:

- a card with no rating;
- a field that changed class name;
- a verification page;
- a page with fewer cards;
- a parse failure.

Do not silently fill missing values before understanding why they are missing.

### 7.9 A cleaned table

| name | rating | reviews_numeric | company_type | hq_city | company_age_years | employees_lower_bound |
|---|---:|---:|---|---|---:|---:|
| TCS | 3.9 | 17,100 | Public | Mumbai | 52 | 10,000 |
| Accenture | 4.0 | 15,000 | Private | Dublin | 31 | 10,000 |
| ICICI Bank | 4.1 | 13,800 | Public | Mumbai | 26 | 10,000 |
| Cognizant | 3.9 | 12,900 | Private | Teaneck | 26 | 10,000 |
| HDFC Bank | 4.0 | 12,000 | Public | Mumbai | 26 | 10,000 |

The values are derived from the saved official first-page outputs.

---

## 8. Results, reproducibility, and interpretation

### 8.1 The official result

| Quantity | Saved value |
|---|---:|
| First-page cards | 30 |
| First-page ratings | 30 |
| First-page review counts | 30 |
| First-page DataFrame shape | (30, 7) |
| Loop pages | 1,000 |
| Final shape | (30,000, 7) |
| Displayed company count | 477,887 |

### 8.2 The subset problem

The official loop collects 30,000 rows. The page displays 477,887 unique companies. Even if the displayed count is approximate, 30,000 is much smaller. The official notebook is a sample of the directory, not a complete crawl.

This matters for analysis:

- do not describe `final` as "all companies";
- do not compute population-level statistics without acknowledging the subset;
- do not compare page 1 rankings with the full directory;
- record the page range and collection date.

### 8.3 The last-page `df`

`df` after the loop is the last page, not the first page and not the full dataset. If you want the first page, keep a separate `first_df` before the loop. If you want the full dataset, use `final`.

### 8.4 No snapshot

The official notebook does not save the data. This is a reproducibility failure. A future run may get different companies, ratings, or a verification page. Save:

```python
final.to_csv("datasets/ambitionbox_companies.csv", index=False)
```

and record:

- URL pattern;
- page range;
- collection date;
- User-Agent;
- number of cards per page;
- number of rows collected;
- any missing fields.

### 8.5 Live pages are not reproducible

The current page may:

- show a verification wall;
- change class names;
- change page size;
- change the number of pages;
- require JavaScript;
- rate-limit the client.

The official saved outputs are evidence of one run, not a guarantee that the same code works today.

### 8.6 What the ratings and reviews mean

- `rating` is a displayed average, rounded to one decimal.
- `reviews` is a displayed count, often rounded (`17.1k`).
- `No_of_Employee` is a bucket (`10000+`), not an exact headcount.
- `Company_Age` is a whole-year display.
- `Head_Quarters` includes a city plus a count of additional locations.

Do not treat rounded display strings as precise measurements. They are features with measurement granularity.

---

## 9. Practical workflow, ethics, and mistakes

### 9.1 A robust scraper pattern

```python
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

base_url = "https://www.ambitionbox.com/list-of-companies"
headers = {"User-Agent": "..."}
frames = []

for page in range(1, 11):
    response = requests.get(
        base_url,
        params={"page": page},
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    cards = soup.find_all("div", class_="companyCardWrapper")

    if not cards:
        print(f"page {page}: no cards; stopping")
        break

    rows = []
    for card in cards:
        name_el = card.find("h2", class_="companyCardWrapper__companyName")
        rating_el = card.find("div", class_="rating_text")
        count_el = card.find(
            "span",
            class_="companyCardWrapper__companyRatingCount",
        )

        rows.append({
            "name": name_el.get_text(strip=True) if name_el else None,
            "rating": rating_el.get_text(strip=True) if rating_el else None,
            "rating_count": count_el.get_text(strip=True) if count_el else None,
        })

    frames.append(pd.DataFrame(rows))
    time.sleep(1)

final = pd.concat(frames, ignore_index=True)
final.to_csv("datasets/ambitionbox_sample.csv", index=False)
```

The pattern adds:

- a timeout;
- status checking;
- a stop condition when no cards are found;
- `None` guards;
- a delay between requests;
- one concatenation;
- a snapshot without the index.

### 9.2 Ethics and legality

Before scraping:

- read the site's terms of service;
- check `robots.txt`;
- prefer an official API or data export;
- do not bypass CAPTCHAs or verification pages;
- do not collect personal data without a lawful basis;
- identify your client honestly where required;
- respect rate limits;
- cache pages instead of re-fetching;
- do not republish copyrighted content wholesale.

A `User-Agent` header is not permission. If the site returns a verification page, stop and reconsider. The 2026-11-04 check returned `Verify - AmbitionBox`; that is a signal, not a puzzle to defeat.

### 9.3 Reliability checklist

- Set a timeout.
- Check `response.status_code`.
- Check for an expected marker or card count.
- Guard every `find`.
- Log page number, status, and card count.
- Stop on an empty page.
- Deduplicate by a stable identifier if available.
- Save raw HTML for debugging if permitted.
- Save a CSV snapshot.
- Record the collection date and page range.

### 9.4 Common mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| No status check | parses a verification page | check status and marker |
| No `None` guard | `AttributeError` | `el.get_text() if el else None` |
| Page-level lists | fields misalign | extract card by card |
| Positional fields | wrong values when a field is missing | find by label/class or validate length |
| Bare `except` | hides bugs | catch specific exceptions |
| `append` in a loop | slow; removed in pandas 2.0 | collect and `pd.concat` |
| No timeout | notebook hangs | pass `timeout` |
| No delay | rate limit or block | sleep between requests |
| No snapshot | data lost on restart | `to_csv(index=False)` |
| Trusting class names | breaks after redesign | validate counts and markers |
| Treating strings as numbers | wrong statistics | convert with `pd.to_numeric` |
| Scraping without permission | legal/ethical risk | read ToS/robots; use an API |

---

## 10. Essential interview and examination preparation

### 10.1 Level 1 - definitions

**Q1. What is web scraping?**

Fetching a web page and extracting structured data from its HTML.

**Q2. What is HTML?**

A markup language that describes the structure and content of a web page.

**Q3. What is a tag?**

An element such as `<div>`, `<h2>`, or `<span>` that marks up content.

**Q4. What is a class?**

An attribute used to group elements for styling and selection.

**Q5. What is BeautifulSoup?**

A Python library that parses HTML/XML into a searchable tree.

### 10.2 Level 2 - mechanics

**Q6. What does `requests.get` return?**

A `Response` object. Use `.text` for HTML or `.content` for bytes.

**Q7. What does `BeautifulSoup(html, "lxml")` do?**

Parses the HTML string into a tree using the `lxml` parser.

**Q8. `find` versus `find_all`?**

`find` returns the first match or `None`; `find_all` returns all matches.

**Q9. Why `class_` instead of `class`?**

`class` is a Python keyword; BeautifulSoup uses `class_` for the attribute.

**Q10. What does `get_text(strip=True)` do?**

Returns descendant text with surrounding whitespace stripped.

**Q11. Why guard `find` with `if el else None`?**

Because `find` can return `None`, and `None.get_text()` raises `AttributeError`.

### 10.3 Level 3 - applied and numerical

**Q12. The official loop is `range(1, 1001)`. How many pages?**

$$
1001-1=1{,}000.
$$

**Q13. Each page has 30 cards. How many rows for 1,000 pages?**

$$
1{,}000 \times 30 = 30{,}000.
$$

**Q14. The saved `final.shape` is (30000, 7). What does that mean?**

30,000 rows and 7 columns.

**Q15. The page displays 477,887 companies. What fraction does 30,000 represent?**

$$
\frac{30{,}000}{477{,}887}\approx 0.06278,
$$

about 6.28%.

**Q16. How many pages would cover 477,887 companies at 30 per page?**

$$
\frac{477{,}887}{30}\approx 15{,}929.57,
$$

so about 15,930 pages.

**Q17. Estimate the row copies from 1,000 repeated appends of 30 rows.**

$$
30 \times \frac{1{,}000 \times 1{,}001}{2}=15{,}015{,}000.
$$

### 10.4 Level 4 - debugging and scenarios

**Q18. `card.find("h2").get_text()` raises `AttributeError`. Why?**

`card.find("h2")` returned `None`; the card has no matching `h2`, or the class changed.

**Q19. `len(company)` is 0. What do you check?**

The status code, the page content, the expected marker, the class name, whether the site returned a verification page, and whether the content is JavaScript-rendered.

**Q20. The first page has 30 cards but the final frame has fewer than 30,000 rows. Why?**

A page failed, returned fewer cards, or the loop appended an empty frame. There is no status check or card-count validation.

**Q21. `df.sample(5)` shows companies that are not on page 1. Why?**

`df` is reassigned inside the loop, so after the loop it holds the last page, not the first page.

**Q22. The CSV has an `Unnamed: 0` column. What happened?**

`to_csv` wrote the index. Use `index=False`.

**Q23. The site returns a 200 page titled `Verify - AmbitionBox`. What does that mean?**

It is a bot-verification page, not the data page. Do not bypass it; use an API, permission, or another source.

### 10.5 Worked code-reading questions

**Q24. What does this return?**

```python
soup.find_all("p", class_="rating")
```

A list-like `ResultSet` of all `p` tags with class `rating`; on the official first page, 30 elements.

**Q25. What is wrong here?**

```python
for card in soup.find_all("div", class_="companyCardWrapper"):
    print(card.find("h2").get_text(strip=True))
```

No `None` guard. If a card lacks an `h2`, it raises `AttributeError`.

**Q26. Why is this inefficient?**

```python
final = pd.DataFrame()
for page in range(1, 1001):
    final = final.append(get_page(page), ignore_index=True)
```

Each append copies the accumulated frame; total copying is quadratic in pages. Use a list and one `pd.concat`.

### 10.6 Exam-style theory with answers

**Question 1.** Why is scraping more fragile than an API?

An API has a documented schema and a stable contract. A web page is designed for humans and can change its layout, class names, pagination, or anti-bot behavior at any time.

**Question 2.** Why is a 200 status not proof of success?

A 200 response can be a verification page, a cookie wall, an error page, or a page with a changed schema. Validate the expected marker and the extracted card count.

**Question 3.** Why is page-level extraction risky?

It creates separate lists for each field. If one card is missing a field, the lists can misalign, so a name may be paired with another company's rating. Card-level extraction keeps fields together.

**Question 4.** Why save a snapshot immediately?

Live pages change. A CSV snapshot makes the analysis reproducible, decouples it from the site, and preserves the data if the notebook or site fails.

**Question 5.** Why convert scraped numbers?

They are strings. A mean of `"3.9"` strings is not a numeric mean; sorting `"17.1k Reviews"` lexicographically is wrong. Convert with explicit parsing and handle missing values.

---

## 11. Revision summary

### 11.1 The pipeline

```text
URL + headers
      |
      v
requests.get(..., timeout=...)
      |
      v
status check + expected marker
      |
      v
BeautifulSoup(html, "lxml")
      |
      v
page -> card -> field
      |
      v
guarded extraction
      |
      v
DataFrame
      |
      v
clean types -> snapshot
```

### 11.2 The three levels of selection

1. page: `soup.find_all("div", class_="...")`
2. card: `card.find("h2", class_="...")`
3. field text: `.get_text(strip=True)`

### 11.3 The three checks before building the DataFrame

1. Did the request succeed?
2. Did the expected marker or card count appear?
3. Did every card yield the fields you need?

### 11.4 The three reproducibility rules

1. Save a CSV snapshot.
2. Record the URL, page range, and date.
3. Do not train a model on a live page.

### 11.5 The one-sentence summary

Fetch HTML, parse it into a tree, narrow from page to card to field, guard every `find`, build a DataFrame, clean the strings, and save a snapshot.

---

# Part B - Advanced Reading and Reference

## 12. HTML and CSS reference

### 12.1 Document structure

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Page title</title>
  </head>
  <body>
    <h1>Heading</h1>
    <p>Paragraph</p>
  </body>
</html>
```

The `<head>` contains metadata; the `<body>` contains visible content.

### 12.2 Common attributes

| Attribute | Meaning |
|---|---|
| `class` | one or more class names |
| `id` | unique identifier |
| `href` | link target |
| `src` | resource URL |
| `data-*` | custom data attributes |
| `aria-*` | accessibility metadata |

### 12.3 CSS selector examples

| Selector | Meaning |
|---|---|
| `div` | all `div` elements |
| `.companyCardWrapper` | elements with that class |
| `#main` | element with id `main` |
| `div.companyCardWrapper` | `div` with that class |
| `div > h2` | direct child `h2` |
| `div h2` | descendant `h2` |
| `a[href]` | links with an `href` |
| `p.rating` | `p` with class `rating` |

### 12.4 Whitespace and text nodes

HTML indentation creates whitespace text nodes. `get_text(strip=True)` removes surrounding whitespace; without it, values may contain newlines and tabs.

### 12.5 Entities

HTML entities such as `&amp;`, `&nbsp;`, and `&#8377;` are decoded by the parser. Scraped text may contain non-breaking spaces that look like normal spaces but compare differently. Normalize with:

```python
text = text.replace("\xa0", " ").strip()
```

### 12.6 Malformed HTML

Real pages are often malformed. Parsers repair missing closing tags differently. `lxml` is lenient and fast; `html5lib` is more browser-like but slower. Parser choice can change the tree for broken markup.

---

## 13. BeautifulSoup API reference

### 13.1 Object types

| Type | Meaning |
|---|---|
| `BeautifulSoup` | the parsed document |
| `Tag` | an element |
| `NavigableString` | text inside a tag |
| `ResultSet` | list-like result of `find_all` |

### 13.2 Search methods

| Method | Returns |
|---|---|
| `find(name, attrs, class_)` | first `Tag` or `None` |
| `find_all(name, attrs, class_, limit)` | `ResultSet` |
| `select(css)` | `ResultSet` |
| `select_one(css)` | first `Tag` or `None` |
| `find_parent` / `find_next_sibling` | navigation |

### 13.3 Reading content

| Expression | Meaning |
|---|---|
| `tag.text` | all descendant text |
| `tag.get_text()` | all descendant text |
| `tag.get_text(strip=True)` | stripped text |
| `tag.string` | single child string or `None` |
| `tag.contents` | children list |
| `tag.children` | child iterator |

### 13.4 Attributes

| Expression | Meaning |
|---|---|
| `tag["href"]` | attribute value; `KeyError` if absent |
| `tag.get("href")` | attribute value or `None` |
| `tag.attrs` | attribute dictionary |
| `tag.get("class")` | list of classes |

### 13.5 Navigation

```python
tag.parent
tag.children
tag.find_next_sibling()
tag.find_previous_sibling()
tag.find_parent()
```

Navigation is useful when the data is relative to a label rather than a stable class.

### 13.6 Parser installation

```bash
pip install beautifulsoup4 lxml
```

`beautifulsoup4` is the package name; `bs4` is the import name.

---

## 14. Table extraction and alternatives

### 14.1 `pandas.read_html`

For genuine `<table>` elements:

```python
tables = pd.read_html(html)
df = tables[0]
```

`read_html` returns a list of DataFrames. It requires a parser such as `lxml` or `html5lib`. It does not help with card layouts such as AmbitionBox, where data is in `div`s, not a table.

### 14.2 Manual table extraction

```python
table = soup.find("table")
headers = [th.get_text(strip=True) for th in table.find_all("th")]
rows = [
    [td.get_text(strip=True) for td in tr.find_all("td")]
    for tr in table.find_all("tr")
    if tr.find_all("td")
]
df = pd.DataFrame(rows, columns=headers)
```

Validate row lengths:

```python
widths = {len(row) for row in rows}
print(widths)
```

If widths differ, `colspan` or malformed rows are present.

### 14.3 `lxml` and XPath

`lxml` can query with XPath:

```python
from lxml import html

tree = html.fromstring(webpage)
names = tree.xpath("//h2[@class='companyCardWrapper__companyName']/text()")
```

XPath is powerful but also brittle. BeautifulSoup is usually easier for beginners.

### 14.4 Selenium and Playwright

Use a browser automation tool when the page renders data with JavaScript and the HTML source does not contain it. These tools are heavier, slower, and harder to run headless, but they execute JavaScript.

### 14.5 Scrapy

Scrapy is a framework for large-scale crawling: request scheduling, concurrency, retries, pipelines, and item definitions. It is overkill for a one-page demo and useful for a maintained crawler.

### 14.6 API discovery

Before scraping, inspect the browser's Network panel. Many sites load data from a JSON endpoint. If that endpoint is public and permitted, use it instead of parsing HTML. This is often more stable and faster.

---

## 15. Pagination, rate limits, and robust scraping

### 15.1 Page-number pagination

```python
for page in range(1, total_pages + 1):
    response = requests.get(base_url, params={"page": page}, timeout=30)
```

Stop when:

- the page has no cards;
- the page number exceeds `total_pages`;
- a `next` link is absent;
- the response is a verification page.

### 15.2 Offset/limit pagination

```python
params = {"offset": offset, "limit": 30}
offset += 30
```

Stop when fewer than `limit` records are returned.

### 15.3 Cursor pagination

```python
cursor = None
while True:
    params = {"limit": 30}
    if cursor:
        params["cursor"] = cursor
    payload = response.json()
    cursor = payload.get("next_cursor")
    if not cursor:
        break
```

### 15.4 Rate limits

A site may return `429 Too Many Requests` or a verification page. Respect `Retry-After`, add delays, and reduce concurrency. Do not rotate proxies or bypass CAPTCHAs to evade limits.

### 15.5 Retries with backoff

```python
for attempt in range(4):
    response = requests.get(url, timeout=30)
    if response.status_code not in (429, 500, 502, 503, 504):
        break
    time.sleep(2 ** attempt)
response.raise_for_status()
```

Retry only transient failures. Do not retry a 404 or a verification page indefinitely.

### 15.6 Logging

Log:

- page number;
- status code;
- response length;
- number of cards found;
- number of rows appended;
- elapsed time.

A scraper without logs is difficult to debug.

### 15.7 Snapshot layout

```text
data/raw/
└── ambitionbox_page_001_2026-11-04.html
data/processed/
└── ambitionbox_companies_2026-11-04.csv
```

Raw HTML preserves evidence; processed CSV is for analysis.

---

## 16. Source-fidelity audit

### 16.1 Official notebook: 33 cells

| Cell | Content | Saved output |
|---:|---|---|
| 0 | imports | - |
| 1 | 403/User-Agent markdown | - |
| 2 | empty | - |
| 3 | request page 1 | - |
| 4 | parse with `lxml` | - |
| 5 | commented `prettify` | - |
| 6 | first `h1` | `'List of companies in India'` |
| 7 | names heading | - |
| 8 | print all `h2` | 30 names |
| 9 | print all `p` | long mixed output |
| 10 | commented loop | - |
| 11 | ratings heading | - |
| 12 | count `p.rating` | 30 |
| 13 | reviews heading | - |
| 14 | count `a.review-count` | 30 |
| 15 | container heading | - |
| 16 | find `company-content-wrapper` | - |
| 17 | count containers | 30 |
| 18 | build first-page DataFrame | - |
| 19 | `name` | 30 names |
| 20 | `rating` | 30 ratings |
| 21 | `reviews` | 30 review strings |
| 22 | `ctype` | 30 types |
| 23 | `hq` | 30 headquarters |
| 24 | `how_old` | 30 ages |
| 25 | `no_of_employee` | 30 employee strings |
| 26 | `df.shape` | `(30, 7)` |
| 27 | all-pages heading | - |
| 28 | 1,000-page loop with `append` | - |
| 29 | `df.sample(5)` | last-page sample |
| 30 | `df.shape` | `(30, 7)` |
| 31 | `final.shape` | `(30000, 7)` |
| 32 | empty | - |

The notebook has no saved status code, no saved CSV, and no saved raw HTML. The first-page outputs and the final shape are the main evidence.

### 16.2 Local notebook: 22 cells

| Cell | Content | Saved output |
|---:|---|---|
| 0 | banner | - |
| 1 | intro | - |
| 2 | parse/select explanation | - |
| 3 | browser header explanation | - |
| 4 | fetch with headers | - |
| 5 | `pip install lxml` (invalid Python) | - |
| 6 | header explanation | - |
| 7 | fetch and parse | - |
| 8 | print `h2` | - |
| 9 | check `rating_text` marker | - |
| 10 | print ratings | - |
| 11 | HTML count example | - |
| 12 | print rating counts | - |
| 13 | zip ratings/counts | - |
| 14 | card explanation | - |
| 15 | fetch and parse again | - |
| 16 | count cards | - |
| 17 | bare `company_cards` expression | - |
| 18 | per-card explanation | - |
| 19 | per-card loop without guards | - |
| 20 | missing-field experiment | - |
| 21 | key takeaways | - |

No code cell has a saved output. The local notebook is a code-reading exercise, not a successful scrape.

### 16.3 Selector drift

| Official | Local |
|---|---|
| `company-content-wrapper` | `companyCardWrapper` |
| `p.rating` | `div.rating_text` |
| `a.review-count` | `span.companyCardWrapper__companyRatingCount` |
| `p.infoEntity` | not used |

The drift is the central lesson: scraping code is coupled to a page's presentation, and presentation changes.

### 16.4 The URL bug

The local URL:

```text
https://www.ambitionbox.com/list-of-companies?campaign=desktop_nav%3E%3Fpage%20%3D4
```

contains an encoded `?page =4` inside the `campaign` parameter. It is not a standard `page=4` query. Use `params={"page": 4}` for clarity.

### 16.5 The live check

A supplementary request on 2026-11-04:

```text
status: 200
length: 20,012 characters
title: Verify - AmbitionBox
checked target classes present: none
```

This is not official notebook output. It shows that a 200 response can be a verification page, and that the saved selectors cannot be assumed to work today.

### 16.6 The missing dataset

The official notebook never writes a CSV. `Day_18/datasets/` is empty. There is no local snapshot to reproduce the 30,000-row result. This is a documented gap, not a dataset to invent.

### 16.7 README discrepancies

The old README referred to:

- `reference_notebooks/reference_notebook.ipynb`, but the actual file is `reference_notebooks/day18.ipynb`;
- a simplified notebook that is not present;
- a primary learner notebook name that is actually `my_notebook.ipynb` plus a removed text dump.

The rewrite corrects the README to the files that actually exist.

---

## 17. Extended comparisons

### 17.1 API versus scraping

| Property | API | Scraping |
|---|---|---|
| Data format | JSON/XML | HTML |
| Schema | documented contract | implicit in markup |
| Stability | higher | lower |
| Auth | explicit | often none or browser-like |
| Pagination | documented | inferred |
| Legal clarity | usually clearer | depends on site/ToS |
| Speed | high | lower |
| Maintenance | lower | higher |

### 17.2 BeautifulSoup versus `pd.read_html`

| Property | BeautifulSoup | `pd.read_html` |
|---|---|---|
| Input | any HTML | `<table>` elements |
| Card layouts | yes | no |
| Selector control | high | low |
| Output | tags/text | DataFrames |
| Use case | custom extraction | simple tables |

### 17.3 `requests` versus Selenium/Playwright

| Property | `requests` | Browser automation |
|---|---|---|
| JavaScript | no | yes |
| Speed | fast | slow |
| Setup | simple | heavier |
| Resource use | low | high |
| Use case | static HTML | dynamic pages |

### 17.4 Page-level versus card-level extraction

| Property | Page-level | Card-level |
|---|---|---|
| Alignment | fragile | robust |
| Missing fields | lists shift | row keeps `None` |
| Code | shorter | slightly longer |
| Debugging | harder | easier |
| Recommended | no | yes |

### 17.5 `append` versus `concat`

| Property | `append` | `concat` |
|---|---|---|
| Availability | removed in pandas 2.0 | current |
| Copy behavior | copies every call | one combine |
| Performance on many pages | poor | good |
| Recommended | no | yes |

---

## 18. Extended interview bank

### Level 1 - Foundations

1. What is HTML?
2. What is a tag?
3. What is a class?
4. What is the DOM?
5. What is web scraping?

### Level 2 - Mechanics

6. What does `requests.get` return?
7. What does BeautifulSoup do?
8. `find` versus `find_all`?
9. Why `class_`?
10. What does `get_text(strip=True)` do?
11. What does `select_one` do?
12. How do you read an attribute?

### Level 3 - Mathematics and applied reasoning

13. `range(1, 1001)` has how many pages?
14. 30 cards per page for 1,000 pages gives how many rows?
15. 30,000 rows over 1,000 pages gives what average?
16. 30,000 out of 477,887 is what fraction?
17. 477,887 at 30 per page needs how many pages?
18. Estimate the row copies from 1,000 appends of 30 rows.

### Level 4 - Debugging and ethics

19. `find` returns `None`. What happens next?
20. The page returns 200 but no cards. What do you check?
21. The site returns a verification page. What do you do?
22. The CSV has an extra index column. What happened?
23. The local URL does not request page 4. Why?
24. The loop collected fewer rows than expected. Why?

### Level 5 - Internals and performance

25. Why is repeated `append` quadratic?
26. What is the difference between `response.text` and `response.content`?
27. How does `lxml` differ from `html.parser`?
28. When would you use `html5lib`?
29. How do you stream or batch a large scrape?
30. How do you avoid loading all pages into memory?

### Level 6 - Design and scenarios

31. Design a scraper with retries and logging.
32. Design a snapshot policy for a daily scrape.
33. How would you test selectors without hitting the live site?
34. How would you detect selector drift?
35. How would you secure headers and cookies?

### Level 7 - Advanced

36. Compare page-number, offset, cursor, and Link-header pagination.
37. How do you handle duplicate records across pages?
38. How do you make a scraper idempotent?
39. How do you reconcile scraped data with an official database?
40. What is a data contract, and why does a scraper need one?

---

## 19. Extended examination questions

### 19.1 Theory

1. Explain the request -> HTML -> tree -> DataFrame pipeline.
2. Explain why scraping is a last resort.
3. Explain the difference between `find` and `find_all`.
4. Explain why a 200 status is not proof of success.
5. Explain why page-level extraction can misalign fields.
6. Explain why scraped numbers need conversion.
7. Explain why a snapshot is necessary.
8. Explain why class names are a fragile contract.

### 19.2 Numerical

9. `range(1, 1001)` has how many values?
10. 1,000 pages at 30 cards per page gives what row count?
11. 30,000 rows over 1,000 pages gives what average?
12. 30,000 / 477,887 is what fraction?
13. 477,887 / 30 is what page count?
14. 30 × (1000 × 1001 / 2) is what?
15. A page has 30 cards and 7 fields. How many values are in the page?
16. A final frame has 30,000 rows and 7 columns. How many values?

### 19.3 Analytical

17. A colleague says "the page returned 200, so the scrape worked." Refute it.
18. A colleague says "`final` contains all 477,887 companies." Correct it.
19. A colleague uses `df.sample(5)` to describe the first page. Correct it.
20. A colleague hard-codes a `User-Agent` to bypass a verification page. Explain the ethical issue.
21. A colleague uses `append` in a new project. Explain the compatibility and performance issues.

---

## 20. References and glossary

### 20.1 Course references

- CampusX Day 18 official notebook: `vendor/100-days-of-machine-learning/day18-pandas-dataframe-using-web-scraping/day18.ipynb`.
- Local reference copy: `reference_notebooks/day18.ipynb`.
- Local practical notebook: `my_notebook.ipynb`.
- No local dataset: the official notebook does not save a CSV.

### 20.2 Technical references

- Python `requests` documentation.
- BeautifulSoup documentation.
- `lxml` documentation.
- pandas `read_html`, `concat`, `to_csv`, and `read_csv` documentation.
- HTML and CSS specifications.
- Robots exclusion protocol.

### 20.3 Glossary

| Term | Meaning |
|---|---|
| BeautifulSoup | Python HTML/XML parser |
| class | HTML attribute used for grouping/selection |
| CSS selector | pattern for selecting elements |
| DOM | tree representation of a document |
| HTML | markup language for web pages |
| `lxml` | fast HTML/XML parser |
| parser | component that turns text into a tree |
| scraping | extracting data from web pages |
| selector drift | class/layout changes that break selectors |
| snapshot | fixed copy of live data |
| tag | HTML element |
| User-Agent | client identifier header |
| verification page | anti-bot page instead of content |

### 20.4 Final checklist

- [ ] I can explain the request -> HTML -> tree -> DataFrame pipeline.
- [ ] I can use `requests.get` with headers, timeout, and status checks.
- [ ] I can parse with BeautifulSoup and select cards.
- [ ] I can guard every `find` against `None`.
- [ ] I can extract fields card by card.
- [ ] I can paginate and stop correctly.
- [ ] I can convert scraped strings into numeric features.
- [ ] I can save a snapshot without an index column.
- [ ] I can explain selector drift and verification pages.
- [ ] I can discuss the ethics and legality of scraping.
