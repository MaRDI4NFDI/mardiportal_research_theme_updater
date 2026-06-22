"""Register the topic-overviews flow as a Prefect deployment on the K8s worker.

Run once (or after changes to parameters/schedule) with:

    PREFECT_API_URL=http://prefect-mardi.zib.de/api \
    PREFECT_API_AUTH_STRING="admin:<password>" \
    python workflow_deploy_kubernetes.py

The schedule can also be set or changed via the Prefect UI after deployment
(Deployments → topic-overviews → + Schedule).
"""
from prefect import flow

if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/MaRDI4NFDI/mardiportal_research_theme_updater.git",
        entrypoint="workflow_main.py:topic_overviews",
    ).deploy(
        name="topic-overviews",
        work_pool_name="K8WorkerPool",
        # Schedule is managed via the Prefect UI (Deployments → + Schedule).
        job_variables={
            "image": "ghcr.io/mardi4nfdi/mardiportal_research_theme_updater:latest",
            "env": {"PREFECT_LOGGING_EXTRA_LOGGERS": "topic_overviews"},
        },
        parameters={
            "since_days": 10,
            "harvest_limit": 0,
            "theme_max_papers": 100,
            "dry_run": False,
            "themes_only": False,
        },
    )
