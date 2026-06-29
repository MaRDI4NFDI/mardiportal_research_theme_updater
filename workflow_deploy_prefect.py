"""Register all topic-overviews Prefect deployments.

Run once (or after changes to parameters/schedule) with:

    PREFECT_API_URL=http://prefect-mardi.zib.de/api \
    PREFECT_API_AUTH_STRING="admin:<password>" \
    python workflow_deploy_prefect.py

Schedules can also be set or changed via the Prefect UI after deployment
(Deployments → <name> → + Schedule).
"""
from prefect import flow

_SOURCE = "https://github.com/MaRDI4NFDI/mardiportal_research_theme_updater.git"
_IMAGE = "ghcr.io/mardi4nfdi/mardiportal_research_theme_updater:latest"
_WORK_POOL = "K8WorkerPool"
_JOB_VARIABLES = {
    "image": _IMAGE,
    "image_pull_policy": "Always",
    "env": {"PREFECT_LOGGING_EXTRA_LOGGERS": "topic_overviews"},
}

if __name__ == "__main__":
    flow.from_source(
        source=_SOURCE,
        entrypoint="workflow_main.py:topic_overviews",
    ).deploy(
        name="topic-overviews",
        work_pool_name=_WORK_POOL,
        job_variables=_JOB_VARIABLES,
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
        parameters={
            "limit": 1000,
            "dry_run": False,
        },
    )
