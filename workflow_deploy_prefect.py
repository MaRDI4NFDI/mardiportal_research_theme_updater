"""Register all topic-overviews Prefect deployments.

Run once (or after changes to parameters/schedule) with:

    PREFECT_API_URL=http://prefect-mardi.zib.de/api \
    PREFECT_API_AUTH_STRING="admin:<password>" \
    python workflow_deploy_prefect.py

Schedules can also be set or changed via the Prefect UI after deployment
(Deployments → <name> → + Schedule).
"""
import os

from prefect import flow

_SOURCE = "https://github.com/MaRDI4NFDI/mardiportal_research_theme_updater.git"
_IMAGE = "ghcr.io/mardi4nfdi/mardiportal_research_theme_updater:latest"
_WORK_POOL = "prod-k8s-pool"
_JOB_VARIABLES = {
    "image": _IMAGE,
    "image_pull_policy": "Always",
    "env": {
        "PREFECT_LOGGING_EXTRA_LOGGERS": "topic_overviews",
        "LAKEFS_URL": "https://lake-bioinfmed.zib.de",
        # Contact for the OpenAlex/Crossref polite pools. Taken from the deploying
        # shell so no address is stored in the repo; empty = no contact sent.
        "TOPIC_OVERVIEWS_OPENALEX_EMAIL": os.environ.get("TOPIC_OVERVIEWS_OPENALEX_EMAIL", ""),
    },
}

if __name__ == "__main__":
    flow.from_source(
        source=_SOURCE,
        entrypoint="workflow_main.py:topic_overviews",
    ).deploy(
        name="topic-overviews",
        work_pool_name=_WORK_POOL,
        job_variables=_JOB_VARIABLES,
        cron="0 1 * * *",  # daily 01:00 UTC; re-deploying without it clears the schedule
        parameters={
            "since_days": 10,
            "harvest_limit": 0,
            "theme_max_papers": 100,
            "dry_run": False,
            "themes_only": False,
        },
    )

    flow.from_source(
        source=_SOURCE,
        entrypoint="workflow_arxiv_update.py:arxiv_update",
    ).deploy(
        name="topic-overviews-arxiv-update",
        work_pool_name=_WORK_POOL,
        job_variables=_JOB_VARIABLES,
        cron="0 2 * * *",  # daily 02:00 UTC, one hour after topic-overviews; re-deploying without it clears the schedule
        parameters={
            "limit": 1000,
            "dry_run": False,
        },
    )
