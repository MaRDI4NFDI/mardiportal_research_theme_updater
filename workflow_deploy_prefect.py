"""Register all topic-overviews Prefect deployments.

Run once (or after changes to parameters/schedule) with:

    PREFECT_API_URL=http://prefect-mardi.zib.de/api \
    PREFECT_API_AUTH_STRING="admin:<password>" \
    python workflow_deploy_prefect.py

Schedules can also be set or changed via the Prefect UI after deployment
(Deployments → <name> → + Schedule).
"""
from prefect.deployments import deploy

from workflow_main import topic_overviews
from workflow_arxiv_update import arxiv_update

_IMAGE = "ghcr.io/mardi4nfdi/mardiportal_research_theme_updater:latest"
_WORK_POOL = "K8WorkerPool"
_JOB_VARIABLES = {"image": _IMAGE, "env": {"PREFECT_LOGGING_EXTRA_LOGGERS": "topic_overviews"}}

if __name__ == "__main__":
    deploy(
        topic_overviews.to_deployment(
            name="topic-overviews",
            job_variables=_JOB_VARIABLES,
            parameters={
                "since_days": 10,
                "harvest_limit": 0,
                "theme_max_papers": 100,
                "dry_run": False,
                "themes_only": False,
            },
        ),
        arxiv_update.to_deployment(
            name="topic-overviews-arxiv-update",
            job_variables=_JOB_VARIABLES,
            parameters={
                "limit": 50,
                "dry_run": False,
            },
        ),
        work_pool_name=_WORK_POOL,
        image=_IMAGE,
        push=False,
    )
