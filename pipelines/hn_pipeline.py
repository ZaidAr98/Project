import dlt
from dotenv import load_dotenv
from dlt.sources.rest_api import RESTAPIConfig, rest_api_resources
from dlt.sources.helpers.rest_client import RESTClient
load_dotenv()

BASE_URL = "https://hn.algolia.com/api/v1/"

WINDOW_START = 1784073600   # 2026-07-15
WINDOW_END   = 1785542400   # 2026-08-01


SURVIVORS_SQL = """
SELECT object_id
FROM stories
WHERE num_comments >= 10
  AND title ILIKE 'Ask HN:%'
  AND title !~* '(who is hiring|who wants to be hired|what are you working on|freelancer|who is quitting)'
ORDER BY num_comments DESC
"""

@dlt.source(name="hn")
def hn_source():
    config: RESTAPIConfig = {
        "client": {
            "base_url": BASE_URL,
            "paginator": "single_page",
        },
        "resource_defaults": {
            "write_disposition": "merge",
        },
        "resources": [
            {
                "name": "stories",
                "primary_key": "objectID",
                "endpoint": {
                    "path": "search_by_date",
                    "data_selector": "hits",
                    "params": {
                        "tags": "ask_hn",
                        "hitsPerPage": 1000,
                        "numericFilters": (
                            f"created_at_i>{WINDOW_START},"
                            f"created_at_i<{WINDOW_END}"
                        ),
                    },
                },
            },
        ],
    }
    yield from rest_api_resources(config)


def surviving_story_ids(pipeline):
    with pipeline.sql_client() as c:
        with c.execute_query(SURVIVORS_SQL) as cur:
            return [row[0] for row in cur.fetchall()]


@dlt.resource(name="comments", primary_key="objectID", write_disposition="merge")
def comments(story_ids):
    client = RESTClient(base_url=BASE_URL)
    for n, sid in enumerate(story_ids, 1):
        hits = client.get(
            "search_by_date",
            params={"tags": f"comment,story_{sid}", "hitsPerPage": 1000},
        ).json()["hits"]

        if not hits:
            raise RuntimeError(f"story {sid} returned 0 comments — tag filter or silent cap")
        if len(hits) == 1000:
            print(f"WARNING: story {sid} truncated at the 1000 cap")

        if n % 25 == 0:
            print(f"{n}/{len(story_ids)} threads")

        for h in hits:
            h.pop("_highlightResult", None)
            yield h


def load():
    pipeline = dlt.pipeline(
        pipeline_name="hn",
        destination="postgres",
        dataset_name="hn_raw",
    )

    info = pipeline.run(hn_source())          
    print(info)

    ids = surviving_story_ids(pipeline)       
    print(f"{len(ids)} threads survived the filters")

    info = pipeline.run(comments(ids))       
    print(info)


if __name__ == "__main__":
    load()