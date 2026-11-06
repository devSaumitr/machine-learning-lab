# Day 17 - Fetching Data from an API

> CampusX 100 Days of Machine Learning, Day 17.
> One unified chapter covering the official TMDb notebook and the separate personal notebook.

### How to read this chapter

**Part A** teaches the problem, HTTP and JSON foundations, the `requests` workflow, the official pagination experiment, the local notebook results, reproducibility, security, and essential interview material. It is sufficient to understand and implement the day without reading Part B.

**Part B** is optional advanced reading: HTTP reference, response internals, pagination and rate-limit patterns, a cell-level source-fidelity audit, extended comparisons, an extended interview bank, and extra examination practice.

The day has no ML model and no train/test split. Its quantitative foundations are **request counts, page counts, row counts, response shapes, and the cost of repeated concatenation**.

---

# Part A - Main Learning Chapter

## 1. The problem: the data lives behind a URL

A CSV is a file you already have. An API is a service you ask for data. The data may change between requests, may be split across pages, may require a key, and may fail for reasons that have nothing to do with your Python code.

Day 17 teaches the basic pipeline:

```text
build request -> HTTP GET -> response object -> JSON -> DataFrame -> inspect -> save snapshot
```

The course demonstrates two APIs:

- The official notebook uses TMDb's top-rated movies endpoint and loops through many pages.
- The local `my_notebook.ipynb` uses JSONPlaceholder users and a JSONFakery movies endpoint.

The pattern is the same in both cases. The details differ, and the details are where bugs live.

### 1.1 Why not just download a CSV?

Because many datasets are only available through an API:

- movie ratings and metadata;
- weather and finance data;
- GitHub issues and repository statistics;
- application analytics;
- inventory and pricing feeds.

An API lets you fetch fresh data programmatically. It also introduces new failure modes: authentication, pagination, rate limits, timeouts, schema drift, and non-reproducible snapshots.

### 1.2 The three-step mental model

```python
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data)
```

This is the core pattern, but the three lines hide important questions:

1. Did the request succeed?
2. Is the response actually JSON?
3. Is `data` the list of rows, or a wrapper around it?
4. Is this one page or the whole dataset?
5. Are the columns flat or nested?
6. Is the result reproducible?

Part A answers these questions in order.

---

## 2. Foundations: HTTP, REST, JSON, and DataFrames

### 2.1 HTTP

HTTP is a request-response protocol. A client sends a request to a server; the server sends a response.

A request has:

- a method (`GET`, `POST`, `PUT`, `PATCH`, `DELETE`, ...);
- a URL;
- headers;
- optional query parameters;
- an optional body.

A response has:

- a status code;
- headers;
- a body, often JSON.

`GET` asks for a resource and should not modify server state. It is the method used by both notebooks.

### 2.2 REST

REST is a style for designing HTTP APIs around resources. A typical REST API has:

- a base URL;
- resource paths such as `/movie/top_rated` or `/users`;
- query parameters such as `page=1`, `language=en-US`, or `api_key=...`;
- JSON representations of resources.

REST is not a protocol and not a standard library. It is a set of conventions. The important practical point is that an API endpoint is a URL with a defined response shape, and that shape can change.

### 2.3 Status codes

| Code | Meaning | Typical handling |
|---:|---|---|
| 200 | OK | parse the body |
| 201 | Created | success for a create request |
| 204 | No Content | success with no body |
| 301/302 | Redirect | `requests` follows by default |
| 400 | Bad Request | fix parameters |
| 401 | Unauthorized | missing or invalid credentials |
| 403 | Forbidden | valid credentials but no permission |
| 404 | Not Found | wrong URL or resource |
| 429 | Too Many Requests | rate limit; wait and retry |
| 500 | Internal Server Error | server problem; retry with backoff |
| 503 | Service Unavailable | temporary outage; retry later |

`response.ok` is `True` for status codes below 400. That is a coarse check. `response.raise_for_status()` raises an exception for 4xx and 5xx, which is usually better than silently parsing an error page.

### 2.4 JSON

The response body is often JSON. `response.json()` parses it into Python objects:

- a JSON object becomes a `dict`;
- a JSON array becomes a `list`;
- nested objects and arrays remain nested.

An API may return a list of records directly:

```json
[
  {"id": 1, "title": "A"},
  {"id": 2, "title": "B"}
]
```

or wrap the records in metadata:

```json
{
  "page": 1,
  "total_pages": 428,
  "results": [
    {"id": 1, "title": "A"},
    {"id": 2, "title": "B"}
  ]
}
```

The wrapper is not the table. You must find the list of records before constructing the DataFrame.

### 2.5 DataFrame

A DataFrame is a two-dimensional table with an index and named columns. A list of flat dictionaries maps naturally to rows and columns:

```python
rows = [
    {"id": 1, "title": "A", "popularity": 10.5},
    {"id": 2, "title": "B", "popularity": 8.2},
]
df = pd.DataFrame(rows)
```

If the dictionaries contain nested dictionaries or lists, the cells remain nested. Use `pd.json_normalize` when you need to flatten them.

### 2.6 The response is not the DataFrame

A common beginner mistake is to treat `response.json()` as if it were already tabular. It is Python data. The DataFrame constructor is a separate step:

```text
HTTP response
   |
   v
response.json()  -> Python list/dict
   |
   v
pd.DataFrame(...) or pd.json_normalize(...)
   |
   v
DataFrame
```

Each arrow can fail. Check the status before parsing, and inspect the Python object before constructing the frame.

---

## 3. The request-response lifecycle

### 3.1 Build the request

A request URL has a base and parameters:

```text
https://api.example.com/movies?page=1&language=en-US
```

With `requests`, pass parameters as a dictionary:

```python
params = {"page": 1, "language": "en-US"}
response = requests.get("https://api.example.com/movies", params=params)
```

`requests` encodes the parameters and appends them to the URL. This is safer than building the query string with an f-string, especially when values contain spaces or special characters.

### 3.2 Send the request

```python
import requests

response = requests.get(url, params=params, timeout=30)
```

`requests.get` returns a `Response` object. It does not return the JSON body directly.

### 3.3 Check the response

```python
response.raise_for_status()
```

This raises an `HTTPError` for 4xx and 5xx. It does not raise for a 200 response whose body is not JSON. For that, `response.json()` will raise a `JSONDecodeError`.

### 3.4 Parse the body

```python
data = response.json()
```

The result is a Python list or dictionary. Print its type and a small sample before indexing:

```python
print(type(data))
print(data if isinstance(data, dict) else data[:2])
```

### 3.5 Find the rows

If `data` is a list, it may already be the rows. If it is a dictionary, find the key that contains the list:

```python
if isinstance(data, dict):
    print(data.keys())
    rows = data["results"]       # example; inspect first
else:
    rows = data
```

Do not guess the key. A `KeyError` is the API telling you that your assumption was wrong.

### 3.6 Build the DataFrame

```python
df = pd.DataFrame(rows)
```

If the records are nested, use `pd.json_normalize(rows)`. If you need only a few columns, select them after construction:

```python
df = pd.DataFrame(rows)[["id", "title", "popularity"]]
```

### 3.7 Save a snapshot

```python
df.to_csv("datasets/movies.csv", index=False)
```

Saving the snapshot separates data collection from analysis. The next run can read the CSV without hitting the API.

---

## 4. `requests` syntax and the `Response` object

### 4.1 `requests.get`

```python
requests.get(
    url,
    params=None,
    headers=None,
    timeout=None,
    auth=None,
    verify=True,
    proxies=None,
)
```

| Argument | Meaning |
|---|---|
| `url` | endpoint |
| `params` | query parameters as a dict |
| `headers` | HTTP headers such as authorization |
| `timeout` | maximum wait in seconds |
| `auth` | authentication tuple or handler |
| `verify` | TLS certificate verification |
| `proxies` | proxy configuration |

The return value is a `Response`, not the data.

### 4.2 Response attributes

| Attribute/method | Meaning |
|---|---|
| `status_code` | integer status |
| `ok` | `True` if status < 400 |
| `raise_for_status()` | raise for 4xx/5xx |
| `json()` | parse body as JSON |
| `text` | decoded body as text |
| `content` | raw bytes |
| `headers` | response headers |
| `url` | final URL after redirects |
| `links` | parsed `Link` header for pagination |
| `elapsed` | time spent |
| `request` | the request that produced the response |

### 4.3 Why `raise_for_status()` first?

Without it, a 401 or 500 response may be parsed as JSON or may raise a confusing decoding error. The actual problem is the HTTP status. Fail loudly at the right layer:

```python
response = requests.get(url, timeout=30)
response.raise_for_status()
data = response.json()
```

### 4.4 Timeouts

A request without a timeout can hang indefinitely. Always set one:

```python
response = requests.get(url, timeout=30)
```

The timeout applies to connection and read operations. For production, use a retry policy with backoff for transient failures.

### 4.5 Headers and authentication

Some APIs use an API key in a query parameter:

```python
params = {"api_key": os.environ["API_KEY"], "page": 1}
```

Others use an authorization header:

```python
headers = {"Authorization": f"Bearer {os.environ['API_TOKEN']}"}
response = requests.get(url, headers=headers, timeout=30)
```

Never hard-code a real key in a notebook you share or commit. The official notebook contains a hard-coded TMDb key; treat that key as exposed and use your own environment variable.

### 4.6 What `response.json()` does not guarantee

- It does not guarantee that the body is JSON; it raises if it is not.
- It does not guarantee that the JSON is the shape you expected.
- It does not guarantee that the request succeeded; check the status first.
- It does not guarantee completeness; the response may be one page.

---

## 5. From JSON to DataFrame

### 5.1 Flat records

```python
rows = [
    {"id": 1, "title": "A", "popularity": 10.5},
    {"id": 2, "title": "B", "popularity": 8.2},
]
df = pd.DataFrame(rows)
```

This is the simplest case. Each dictionary becomes a row; each key becomes a column.

### 5.2 Nested records

If a record contains a nested object:

```python
rows = [
    {"id": 1, "title": "A", "metadata": {"year": 2020, "runtime": 120}},
]
```

`pd.DataFrame(rows)` keeps `metadata` as a dictionary in one cell. Use:

```python
df = pd.json_normalize(rows)
```

to get `metadata.year` and `metadata.runtime` as columns. Day 16 covers this in detail.

### 5.3 Selecting columns

Two equivalent patterns:

```python
# Select after constructing the DataFrame
df = pd.DataFrame(rows)[["id", "title", "popularity"]]

# Select from the raw records before constructing
df = pd.DataFrame([
    {"id": row["id"], "title": row["title"], "popularity": row["popularity"]}
    for row in rows
])
```

The first is shorter and usually preferred. If the API returns many fields, selecting after construction still parses everything; selecting before construction can reduce memory only if you discard the raw objects early.

### 5.4 Missing keys

`pd.DataFrame(rows)` tolerates missing keys by inserting `NaN` for the missing columns. That can be useful, but it can also hide schema drift. Inspect `df.isna().sum()` and compare the columns to the expected schema.

### 5.5 DataFrame versus `json_normalize`

| Situation | Use |
|---|---|
| flat list of dictionaries | `pd.DataFrame(rows)` |
| nested dictionaries | `pd.json_normalize(rows)` |
| nested list of records | `pd.json_normalize(rows, record_path=...)` |
| JSON string in a column | `json.loads` then normalize |

---

## 6. The official notebook: TMDb pagination

### 6.1 The endpoint

The official notebook requests TMDb's top-rated movies:

```python
response = requests.get(
    "https://api.themoviedb.org/3/movie/top_rated"
    "?api_key=...&language=en-US&page=1"
)
```

The response contains a `results` list. Each result has fields such as `id`, `title`, `overview`, `release_date`, `popularity`, `vote_average`, and `vote_count`.

### 6.2 The first page

```python
temp_df = pd.DataFrame(response.json()["results"])[
    ["id", "title", "overview", "release_date",
     "popularity", "vote_average", "vote_count"]
]
```

The first page contains 20 results, consistent with the final row count. TMDb's standard page size is 20, and the selected frame has 7 columns.

### 6.3 The saved `df.head()` anomaly

The visible notebook code creates `temp_df`, then calls:

```python
df.head()
```

before `df` is defined. The saved output shows movie rows, so the output came from a previous `df` in the kernel state. The execution counts confirm out-of-order execution: the `df.head()` cell has a later execution count than the cell that resets `df = pd.DataFrame()`.

This is a reproducibility lesson: a notebook's saved outputs are not proof that the visible cells run top-to-bottom from a fresh kernel.

### 6.4 The pagination loop

```python
df = pd.DataFrame()

for i in range(1, 429):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/top_rated"
        "?api_key=...&language=en-US&page={}".format(i)
    )
    temp_df = pd.DataFrame(response.json()["results"])[
        ["id", "title", "overview", "release_date",
         "popularity", "vote_average", "vote_count"]
    ]
    df = df.append(temp_df, ignore_index=True)
```

`range(1, 429)` produces 428 page numbers: 1 through 428. The saved final frame has 8,551 rows and 7 columns.

### 6.5 The row-count arithmetic

TMDb returns 20 rows per page. If all 428 pages were full:

$$
428 \times 20 = 8{,}560.
$$

The saved frame has 8,551 rows, nine fewer. A consistent decomposition is:

$$
427 \times 20 + 11 = 8{,}551.
$$

So the first 427 pages contributed 20 rows each, and the final page contributed 11. The average rows per page is

$$
\frac{8{,}551}{428}\approx 19.98.
$$

This arithmetic is a useful check: if the final frame had 8,560 rows, the last page would have been full; if it had far fewer, the loop or the API response would be suspect.

### 6.6 The `append` problem

`df.append(temp_df, ignore_index=True)` was removed from pandas 2.0. The official notebook was executed with an older pandas. In pandas 3.0.5, the loop would raise `AttributeError`.

The modern replacement is to collect the frames and concatenate once:

```python
frames = []

for page in range(1, 429):
    response = requests.get(url, params={"page": page}, timeout=30)
    response.raise_for_status()
    rows = response.json()["results"]
    frames.append(pd.DataFrame(rows)[columns])

df = pd.concat(frames, ignore_index=True)
```

### 6.7 Why repeated `append` is expensive

Each `append` creates a new DataFrame and copies the accumulated data. If page $k$ adds 20 rows, the total number of row copies is roughly

$$
20 + 40 + \cdots + 8{,}551
= \frac{428 \times 8{,}551}{2}
\approx 1{,}829{,}914.
$$

That is about 1.83 million row copies for a final frame of 8,551 rows. Collecting frames in a list and concatenating once avoids most of that copying. This is the algorithmic reason the pandas developers removed `append`.

### 6.8 The saved CSV

The official notebook ends with:

```python
df.to_csv("movies.csv")
```

The default `to_csv` writes the index. The resulting `datasets/movies.csv` has 8,551 rows and 8 columns:

```text
Unnamed: 0, id, title, overview, release_date, popularity, vote_average, vote_count
```

The `Unnamed: 0` column is the old integer index. If you read it back:

```python
df = pd.read_csv("datasets/movies.csv", index_col=0)
```

or save correctly in the first place:

```python
df.to_csv("datasets/movies.csv", index=False)
```

The difference between 7 columns in memory and 8 columns in the CSV is a serialization artifact, not a new feature.

### 6.9 What the official notebook does not do

- It does not check `response.status_code`.
- It does not call `raise_for_status()`.
- It does not set a timeout.
- It does not handle a failed page.
- It does not sleep between requests.
- It does not deduplicate movie IDs.
- It does not save the raw JSON.
- It hard-codes an API key.
- It uses `append`, which is removed in modern pandas.
- It writes the index to CSV.

These are teaching gaps, not reasons to copy the notebook into production.

---

## 7. The local notebook: two APIs

### 7.1 JSONPlaceholder users

```python
url = "https://jsonplaceholder.typicode.com/users"
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data)
print(df)
```

The users endpoint returns a JSON array, so `pd.DataFrame(data)` works. Each user record contains nested `address`, `company`, and other fields. If displayed, those cells would remain dictionaries. The saved notebook does not show the output, so the first fetch is not inspectable from the saved file.

### 7.2 JSONFakery movies

```python
url = "https://jsonfakery.com/movies/simple-paginate"
rs = requests.get(url)
result = rs.json()

pd.DataFrame(result["data"])[["original_title", "popularity"]]
```

The saved output has 20 rows and 2 columns. The API wraps the records under `data`. The notebook selects only `original_title` and `popularity`.

The saved values include movies such as:

| original_title | popularity |
|---|---:|
| Pet Sematary: Bloodlines | 368.52 |
| Lambert the Sheepish Lion | 7.98 |
| Lilo & Stitch 2: Stitch Has a Glitch | 25.61 |

These are snapshot values. The live API can return different rows or values on the next call.

### 7.3 What the local notebook demonstrates

- A list response can go straight into `pd.DataFrame`.
- A wrapped response needs `result["data"]`.
- Column selection is a separate step.
- A live API result is not reproducible without a snapshot.

### 7.4 What the local notebook does not do

- It does not check the status before `.json()`.
- It does not call `raise_for_status()`.
- It does not set a timeout.
- It does not inspect `result.keys()` before using `result["data"]`.
- It does not paginate the movies endpoint.
- It does not save the raw response.
- It does not display the first users response.

---

## 8. Results, reproducibility, and interpretation

### 8.1 The two experiments are different

| Experiment | Endpoint | Rows | Columns | Pages |
|---|---|---:|---:|---:|
| Official TMDb | top-rated movies | 8,551 | 7 | 428 |
| Local JSONFakery | simple-paginate movies | 20 | 2 | 1 |
| Local JSONPlaceholder | users | not displayed | not displayed | 1 |

Do not compare the 20-row local movie result with the 8,551-row official result as if they were the same dataset. They are different APIs, different fields, and different collection scopes.

### 8.2 Live data is not reproducible

An API response is a snapshot of a changing system. The same URL can return different data tomorrow. For experiments:

1. save the raw JSON or the resulting CSV;
2. record the URL, query parameters, and timestamp;
3. use the saved snapshot for model training and evaluation;
4. refresh the snapshot deliberately, not accidentally.

The `movies.csv` file is such a snapshot. It is useful precisely because it does not change.

### 8.3 The `Unnamed: 0` column

The saved CSV has an extra index column. This is a common source of confusion:

```python
df = pd.read_csv("movies.csv")
print(df.columns)
# ['Unnamed: 0', 'id', 'title', ...]
```

Fix it at read time:

```python
df = pd.read_csv("movies.csv", index_col=0)
```

or at write time:

```python
df.to_csv("movies.csv", index=False)
```

Do not treat the unnamed column as a feature.

### 8.4 Data types

The saved `movies.csv` has:

```text
id             int64
title          str
overview       str
release_date   str
popularity     float64
vote_average   float64
vote_count     int64
```

`release_date` is text, not a datetime. Convert it when you need date arithmetic:

```python
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
```

`overview` is long text and may contain commas, quotes, or newlines. CSV quoting handles this, but naive splitting on commas does not.

### 8.5 What the numbers mean

- `popularity` is a TMDb-specific score, not a universal measure.
- `vote_average` is an average rating on a changing scale.
- `vote_count` is the number of votes at snapshot time.
- `id` is the stable movie identifier within TMDb.

A high popularity value does not imply a high rating, and a high rating with very few votes is uncertain. These are analysis questions for later days, not properties guaranteed by the API.

---

## 9. Practical workflow, security, and mistakes

### 9.1 A robust fetch pattern

```python
import os
import time
import requests
import pandas as pd

url = "https://api.example.com/movies"
headers = {"Authorization": f"Bearer {os.environ['API_TOKEN']}"}
frames = []

for page in range(1, 4):
    response = requests.get(
        url,
        params={"page": page},
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    rows = payload["results"]
    if not rows:
        break

    frames.append(pd.DataFrame(rows))
    time.sleep(0.5)

df = pd.concat(frames, ignore_index=True)
df.to_csv("datasets/movies_snapshot.csv", index=False)
```

The pattern adds:

- environment-based credentials;
- status checking;
- a timeout;
- an empty-page stop condition;
- a delay between requests;
- one concatenation;
- a snapshot without the index.

### 9.2 Security checklist

- Never commit API keys, tokens, or passwords.
- Read secrets from environment variables or a secrets manager.
- Treat a key embedded in a shared notebook as exposed; rotate it.
- Use HTTPS.
- Do not log full response headers if they contain tokens.
- Parameterize URLs with `params`, not string concatenation.
- Validate downloaded data before using it.

### 9.3 Reliability checklist

- Set a timeout.
- Handle `401`, `403`, `429`, and `5xx`.
- Retry transient failures with backoff.
- Respect `Retry-After`.
- Stop pagination on an empty page or the last page.
- Deduplicate by a stable ID if pages can shift.
- Save the raw response for debugging.

### 9.4 Common mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Calling `.json()` before checking status | confusing parse error | `raise_for_status()` first |
| Assuming the top-level key | `KeyError` | print `data.keys()` |
| Treating one page as the full dataset | too few rows | paginate |
| No timeout | notebook hangs | pass `timeout` |
| Hard-coded key | secret leak | environment variable |
| `to_csv` with index | `Unnamed: 0` | `index=False` |
| Repeated `append` | slow loop, removed API | collect and `pd.concat` |
| Live data in a model experiment | unreproducible results | save a snapshot |
| Comparing different APIs | false conclusions | compare like with like |
| Ignoring rate limits | 429 errors | delay and backoff |

---

## 10. Essential interview and examination preparation

### 10.1 Level 1 - definitions

**Q1. What is an API?**

An application programming interface. In this day's context, an HTTP service that returns data, usually JSON, in response to a request.

**Q2. What is a GET request?**

An HTTP method that asks for a resource without modifying server state.

**Q3. What is a status code?**

A three-digit HTTP response code indicating success, redirection, client error, or server error.

**Q4. What is JSON?**

A text format for structured data built from objects, arrays, strings, numbers, booleans, and null.

### 10.2 Level 2 - mechanics

**Q5. What does `requests.get` return?**

A `Response` object, not the JSON body.

**Q6. What does `response.json()` do?**

Parses the response body as JSON and returns Python objects.

**Q7. Why call `raise_for_status()`?**

To turn 4xx and 5xx responses into exceptions before parsing the body.

**Q8. What does `params` do?**

It encodes a dictionary as URL query parameters.

**Q9. Why set a timeout?**

To prevent the program from waiting indefinitely on a slow or dead server.

**Q10. `pd.DataFrame(rows)` versus `pd.json_normalize(rows)`?**

Use `DataFrame` for flat records; use `json_normalize` for nested dictionaries or lists.

### 10.3 Level 3 - applied and numerical

**Q11. `range(1, 429)` iterates how many pages?**

$$
429-1=428.
$$

**Q12. TMDb returns 20 rows per page. If 428 pages are full, how many rows?**

$$
428 \times 20 = 8{,}560.
$$

**Q13. The saved frame has 8,551 rows. How many rows are missing from the full-page ceiling?**

$$
8{,}560-8{,}551=9.
$$

A consistent decomposition is $427 \times 20 + 11 = 8{,}551$.

**Q14. What is the average rows per page?**

$$
\frac{8{,}551}{428}\approx 19.98.
$$

**Q15. Why does `movies.csv` have 8 columns when the DataFrame has 7?**

Because `to_csv` wrote the index. The extra `Unnamed: 0` column is the index, not a feature.

**Q16. Approximately how many row copies does repeated `append` cause?**

$$
20+40+\cdots+8{,}551
=\frac{428 \times 8{,}551}{2}
\approx 1{,}829{,}914.
$$

### 10.4 Level 4 - debugging and scenarios

**Q17. `response.json()` raises `JSONDecodeError`. What are the likely causes?**

The body is not JSON, the request returned an HTML error page, or the endpoint changed. Check the status code and `response.text` first.

**Q18. `result["data"]` raises `KeyError`. What do you do?**

Print `result.keys()` and inspect the actual envelope. Do not guess the key.

**Q19. A notebook works today but fails tomorrow because a column disappeared. What is the process fix?**

Save a raw snapshot, validate the schema, and treat the API as an external contract that can change. Add a schema check before selecting columns.

**Q20. A loop over 428 pages is slow and memory-heavy. What do you change?**

Use one `pd.concat` instead of repeated `append`; add a timeout, retries, and a delay; consider streaming to disk or processing page by page.

**Q21. The CSV has an `Unnamed: 0` column. What happened?**

`to_csv` wrote the index. Use `index=False` when saving, or `index_col=0` when reading.

### 10.5 Worked code-reading questions

**Q22. What is the shape of the result?**

```python
rows = [
    {"id": 1, "title": "A"},
    {"id": 2, "title": "B"},
]
pd.DataFrame(rows)
```

Shape `(2, 2)`.

**Q23. What is wrong with this code?**

```python
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data["results"])
```

It does not check the status, does not set a timeout, and assumes the `results` key. It may fail or parse an error body.

**Q24. Why is this loop inefficient?**

```python
df = pd.DataFrame()
for page in range(1, 429):
    df = df.append(get_page(page), ignore_index=True)
```

Each append copies the accumulated frame. The total copying is quadratic in the number of pages. Use a list and one `pd.concat`.

### 10.6 Exam-style theory with answers

**Question 1.** Explain why `response.ok` is not enough to guarantee usable data.

`response.ok` only says the status is below 400. A 200 response can contain an HTML page, an empty body, a JSON error object, or a different schema. You still need to parse and validate the body.

**Question 2.** Why is a live API a reproducibility problem?

The response can change between calls. A model trained on today's data may not be comparable with one trained on tomorrow's. Save a snapshot and record the request metadata.

**Question 3.** Why is hard-coding an API key dangerous even in a course notebook?

Notebooks are shared, copied, committed, and posted. A key in a notebook can be harvested. Use environment variables and rotate exposed keys.

**Question 4.** Why does `df.append` have a different cost profile from `pd.concat`?

`append` creates a new frame for each call and copies the existing data. `concat` can combine all frames in one operation. For many pages, repeated copying dominates.

**Question 5.** A colleague says "the API returned 20 rows, so the dataset has 20 rows." Correct the statement.

The response is one page. The dataset may have many pages. The 20 rows are the page size, not the dataset size. Pagination metadata or a loop is required to know the total.

---

## 11. Revision summary

### 11.1 The pipeline

```text
build URL + params
      |
      v
requests.get(url, params=..., timeout=...)
      |
      v
response.raise_for_status()
      |
      v
response.json()
      |
      v
find the row list
      |
      v
pd.DataFrame(rows) or pd.json_normalize(rows)
      |
      v
select columns / validate / save snapshot
```

### 11.2 The three checks before parsing

1. Is the status successful?
2. Is the body JSON?
3. Is the top-level object a list or a wrapper dictionary?

### 11.3 The three reproducibility rules

1. Save the raw response or a CSV snapshot.
2. Record the URL, parameters, and timestamp.
3. Never train a model on a moving target.

### 11.4 The three common bugs

- missing status check;
- guessing the envelope key;
- treating one page as the whole dataset.

### 11.5 The one-sentence summary

Use `requests` to fetch, check the status, parse the JSON, find the row list, build a DataFrame, and save a snapshot before analysis.

---

# Part B - Advanced Reading and Reference

## 12. HTTP and REST reference

### 12.1 Request anatomy

```text
GET /movie/top_rated?page=1&language=en-US HTTP/1.1
Host: api.themoviedb.org
Authorization: Bearer <token>
Accept: application/json
```

- method: `GET`;
- path: `/movie/top_rated`;
- query: `page=1&language=en-US`;
- headers: host, authorization, accept.

### 12.2 Methods

| Method | Meaning | Safe | Idempotent |
|---|---|---|---|
| GET | read | yes | yes |
| POST | create | no | no |
| PUT | replace | no | yes |
| PATCH | partial update | no | no |
| DELETE | remove | no | yes |

"Safe" means no intended state change. "Idempotent" means repeating the request has the same effect.

### 12.3 Status-code families

| Family | Meaning |
|---|---|
| 2xx | success |
| 3xx | redirection |
| 4xx | client error |
| 5xx | server error |

### 12.4 Content negotiation

The `Accept` header tells the server which representation you want. `Accept: application/json` is common. The response `Content-Type` header tells you what was actually returned.

### 12.5 Pagination styles

| Style | Example | How to continue |
|---|---|---|
| page number | `page=2` | increment until `total_pages` |
| offset/limit | `offset=100&limit=20` | add limit to offset |
| cursor | `cursor=abc` | follow the returned cursor |
| Link header | `Link: <...>; rel="next"` | follow `response.links["next"]` |

### 12.6 Rate limiting

A server may return `429 Too Many Requests` and a `Retry-After` header. A robust client waits, retries with backoff, and does not hammer the endpoint. Rate limits are part of the API contract.

---

## 13. `requests` and `Response` reference

### 13.1 Common `requests` arguments

| Argument | Purpose |
|---|---|
| `params` | query parameters |
| `headers` | request headers |
| `data` | form-encoded body |
| `json` | JSON body |
| `timeout` | connection/read timeout |
| `auth` | basic/digest authentication |
| `verify` | TLS verification |
| `allow_redirects` | follow redirects |
| `stream` | stream the response body |

### 13.2 `Response` methods

| Method/attribute | Returns |
|---|---|
| `status_code` | int |
| `ok` | bool |
| `raise_for_status()` | raises `HTTPError` or returns `None` |
| `json()` | Python object |
| `text` | str |
| `content` | bytes |
| `headers` | case-insensitive dict |
| `links` | dict of pagination links |
| `elapsed` | timedelta |
| `url` | final URL |

### 13.3 Error classes

- `requests.exceptions.Timeout`
- `requests.exceptions.ConnectionError`
- `requests.exceptions.HTTPError`
- `requests.exceptions.RequestException` (base class)

A production fetch wraps the request in a `try/except` and decides whether to retry, fail, or skip.

### 13.4 JSON decoding

`response.json()` uses the response body and the JSON decoder. If the body is empty, the result may raise or return an empty value depending on the version. Check the status and the body length before parsing.

---

## 14. Pagination and rate-limit patterns

### 14.1 Page-number loop

```python
frames = []
page = 1

while True:
    response = requests.get(url, params={"page": page}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    rows = payload["results"]

    if not rows:
        break

    frames.append(pd.DataFrame(rows))
    page += 1

    if page > payload.get("total_pages", page):
        break

    time.sleep(0.5)

df = pd.concat(frames, ignore_index=True)
```

The loop stops on an empty page or when the page exceeds `total_pages`.

### 14.2 Cursor loop

```python
cursor = None
frames = []

while True:
    params = {"limit": 100}
    if cursor:
        params["cursor"] = cursor

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    frames.append(pd.DataFrame(payload["results"]))
    cursor = payload.get("next_cursor")
    if not cursor:
        break

df = pd.concat(frames, ignore_index=True)
```

Cursors are common in APIs that cannot guarantee stable page numbers.

### 14.3 Deduplication

If pages can shift while you paginate, the same record may appear twice. Deduplicate on a stable ID:

```python
df = df.drop_duplicates(subset=["id"])
```

Do not deduplicate on the title alone; titles are not guaranteed unique.

### 14.4 Retry with backoff

```python
import time

for attempt in range(5):
    response = requests.get(url, timeout=30)
    if response.status_code != 429:
        break
    wait = int(response.headers.get("Retry-After", 2 ** attempt))
    time.sleep(wait)

response.raise_for_status()
```

Exponential backoff avoids overwhelming a struggling server.

### 14.5 Snapshot layout

A reproducible ingestion folder might look like:

```text
data/raw/
└── tmdb_top_rated_2026-11-04.json
data/processed/
└── movies_2026-11-04.csv
```

The date in the filename is part of the provenance. The analysis reads the processed file, not the live API.

---

## 15. Source-fidelity audit

### 15.1 Official notebook: 11 cells

| Cell | Content | Saved output |
|---:|---|---|
| 0 | import pandas, requests | - |
| 1 | request TMDb page 1 | - |
| 2 | build `temp_df` from `results` | - |
| 3 | `df.head()` | movie rows, stale state |
| 4 | `df = pd.DataFrame()` | - |
| 5 | display empty `df` | empty frame |
| 6 | loop pages 1-428 with `append` | - |
| 7 | display `df` | 8,551 rows |
| 8 | `df.shape` | `(8551, 7)` |
| 9 | `df.to_csv("movies.csv")` | writes index |
| 10 | empty | - |

The execution counts are out of order: the `df.head()` cell ran after the reset cell. The saved output is not a top-to-bottom fresh-kernel run.

### 15.2 Local notebook: 9 cells

| Cell | Content | Saved output |
|---:|---|---|
| 0 | documentation banner | - |
| 1 | section heading | - |
| 2 | `requests.get` explanation | - |
| 3 | JSONPlaceholder users fetch and print | no saved output |
| 4 | movies API example | - |
| 5 | column-selection explanation | - |
| 6 | JSONFakery fetch and select | 20 rows x 2 |
| 7 | experiment prompt | - |
| 8 | key takeaways | - |

The first fetch is not displayed. The second fetch has no status check, timeout, or pagination.

### 15.3 The legacy text dump

`DAY17_Fetching-Data-From-API.ipynb.txt` is a plain-text dump of the local notebook, not a notebook. It contains no unique code beyond the nine cells above. It is a candidate for cleanup, not a second source.

### 15.4 The dataset

`datasets/movies.csv` is the saved TMDb snapshot. It has 8,551 rows and 8 columns because the index was written. The in-memory official frame has 7 columns. The extra column is `Unnamed: 0`.

### 15.5 README discrepancies

The old README referred to:

- `reference_notebooks/reference_notebook.ipynb`, but the actual file is `reference_notebooks/day17.ipynb`;
- a simplified notebook that is not present;
- a primary learner notebook name that is actually `my_notebook.ipynb` plus the text dump.

The rewrite corrects the README to the files that actually exist.

---

## 16. Extended comparisons

### 16.1 API versus CSV versus database

| Property | API | CSV | Database |
|---|---|---|---|
| Freshness | live | snapshot | live |
| Reproducibility | low without snapshot | high | depends on snapshot |
| Schema | external contract | inferred | defined |
| Pagination | common | no | query/limit |
| Auth | common | no | common |
| Nested data | common | no | normalized |

### 16.2 `pd.DataFrame` versus `json_normalize`

| Property | `pd.DataFrame` | `json_normalize` |
|---|---|---|
| Flat records | good | good |
| Nested dicts | leaves dict cells | flattens |
| Nested lists | leaves list cells | can use `record_path` |
| Parent metadata | not repeated | `meta` |

### 16.3 `append` versus `concat`

| Property | `append` | `concat` |
|---|---|---|
| Availability | removed in pandas 2.0 | current |
| Copy behavior | copies on every call | one combine |
| Performance on many pages | poor | good |
| Recommended | no | yes |

### 16.4 Live fetch versus snapshot

| Property | Live fetch | Snapshot |
|---|---|---|
| Values | change | fixed |
| Reproducible | no | yes |
| Network required | yes | no |
| Debuggable later | difficult | easy |
| Production use | refresh job | analysis/training |

---

## 17. Extended interview bank

### Level 1 - Foundations

1. What is HTTP?
2. What is a REST API?
3. What is a status code?
4. What is JSON?
5. What is an endpoint?

### Level 2 - Mechanics

6. What does `requests.get` return?
7. What does `response.json()` do?
8. What does `raise_for_status()` do?
9. What does `params` do?
10. Why set a timeout?
11. What is a response header?
12. What is a query parameter?

### Level 3 - Mathematics and applied reasoning

13. How many pages does `range(1, 429)` iterate?
14. If 428 pages have 20 rows each, what is the ceiling?
15. If the actual frame has 8,551 rows, how many rows are missing from the ceiling?
16. What is the average rows per page?
17. Why does the CSV have one more column than the DataFrame?
18. Estimate the row copies caused by 428 repeated appends.

### Level 4 - Debugging and leakage

19. `response.json()` fails. What do you check?
20. `result["data"]` raises `KeyError`. What do you check?
21. The API returns 429. What do you do?
22. The CSV has `Unnamed: 0`. What happened?
23. The notebook works interactively but fails on a fresh run. Why?
24. The API returns fewer rows than expected. What are the possible causes?

### Level 5 - Internals and performance

25. Why is repeated `append` quadratic?
26. How does `requests` handle redirects?
27. What is the difference between `response.text` and `response.content`?
28. What is connection pooling?
29. When would you stream a response?
30. How do you avoid loading all pages into memory?

### Level 6 - Design and scenarios

31. Design a paginated ingestion job with retries.
32. Design a snapshot policy for a daily API.
33. How would you test an API client without hitting the live service?
34. How would you detect schema drift?
35. How would you secure API credentials in a notebook workflow?

### Level 7 - Advanced

36. Compare page-number, offset, cursor, and Link-header pagination.
37. How do you handle duplicate records across pages?
38. How do you make an API ingestion job idempotent?
39. How do you reconcile API data with a database snapshot?
40. What is a data contract, and why does an API need one?

---

## 18. Extended examination questions

### 18.1 Theory

1. Explain the request-response model.
2. Explain the difference between `GET` and `POST`.
3. Explain why status codes matter before parsing.
4. Explain the role of `timeout`.
5. Explain why live API data is not reproducible.
6. Explain why `to_csv(index=False)` matters.
7. Explain why `append` was removed from pandas.
8. Explain the difference between a page and a dataset.

### 18.2 Numerical

9. `range(1, 429)` has how many values?
10. 428 pages at 20 rows per page gives what ceiling?
11. 8,551 rows over 428 pages gives what average?
12. A CSV has 8,551 rows and 8 columns; the frame has 7 columns. What is the extra column?
13. A loop appends 428 pages of 20 rows. Estimate the total row copies.
14. A page returns 11 rows. How many rows are in 427 full pages plus that page?
15. A response has 20 rows and 7 fields. How many values are in the page?

### 18.3 Analytical

16. A colleague claims that `response.ok` guarantees JSON. Refute it.
17. A colleague claims that one page is the dataset. Correct it.
18. A colleague hard-codes an API key "because it is only a course notebook." Explain the risk.
19. A colleague compares a 20-row local result with an 8,551-row official result. Explain why that is invalid.
20. A colleague uses `df.append` in a new project. Explain the compatibility and performance issues.

---

## 19. References and glossary

### 19.1 Course references

- CampusX Day 17 official notebook: `vendor/100-days-of-machine-learning/day17-api-to-dataframe/day17.ipynb`.
- Local reference copy: `reference_notebooks/day17.ipynb`.
- Local practical notebook: `my_notebook.ipynb`.
- Saved snapshot: `datasets/movies.csv`.

### 19.2 Technical references

- Python `requests` documentation.
- HTTP specification and status-code registry.
- TMDb API documentation.
- JSONPlaceholder and JSONFakery API documentation.
- pandas `DataFrame`, `concat`, `to_csv`, and `read_csv` documentation.

### 19.3 Glossary

| Term | Meaning |
|---|---|
| API | application programming interface |
| endpoint | URL that exposes a resource |
| GET | HTTP method for reading |
| header | metadata sent with a request or response |
| JSON | text format for structured data |
| pagination | splitting a large result into pages |
| query parameter | key-value pair in a URL query string |
| rate limit | server restriction on request frequency |
| response | server's reply to a request |
| snapshot | fixed copy of live data |
| status code | numeric HTTP result code |
| timeout | maximum wait for a request |

### 19.4 Final checklist

- [ ] I can explain the GET -> JSON -> DataFrame pipeline.
- [ ] I can use `requests.get` with `params`, `headers`, and `timeout`.
- [ ] I can check status with `raise_for_status()`.
- [ ] I can inspect the JSON envelope before indexing.
- [ ] I can paginate and stop correctly.
- [ ] I can avoid repeated `append`.
- [ ] I can save a snapshot without an index column.
- [ ] I can explain why live API data is not reproducible.
- [ ] I can keep API keys out of notebooks.
- [ ] I can validate the resulting DataFrame before analysis.
