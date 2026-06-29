"""Upload paper Markdown to lakeFS using 2-2-2 QID sharding."""
from __future__ import annotations

import lakefs


def shard_qid(qid: str) -> str:
    """Return the sharded directory prefix for a QID using 2-2-2 zero-padding.

    Example: "Q6190920" → "61/90/92/Q6190920"
    """
    normalized = qid.upper()
    if not normalized.startswith("Q"):
        raise ValueError(f"QID must start with 'Q', got: {qid!r}")
    digits = normalized[1:]
    if not digits.isdigit():
        raise ValueError(f"QID must contain digits after 'Q', got: {qid!r}")
    padded = digits.zfill(6)
    return f"{padded[0:2]}/{padded[2:4]}/{padded[4:6]}/{normalized}"


def component_path(qid: str) -> str:
    """Return the branch-relative lakeFS path for the arxiv fulltext component."""
    normalized = qid.upper()
    return f"{shard_qid(normalized)}/fulltext/{normalized}.md"


def upload_markdown(
    qid: str,
    markdown: str,
    *,
    url: str,
    user: str,
    password: str,
    repo: str,
    branch: str = "main",
) -> str:
    """Upload *markdown* to lakeFS and return the full object path (branch/...).

    Does not commit — call commit_upload() afterwards.
    """
    client = lakefs.Client(host=url, username=user, password=password)
    path = component_path(qid)
    lakefs.repository(repo, client=client).branch(branch).object(path).upload(
        markdown.encode("utf-8"),
        mode="wb",
        content_type="text/markdown; charset=utf-8",
    )
    return f"{branch}/{path}"


def commit_upload(
    message: str,
    *,
    url: str,
    user: str,
    password: str,
    repo: str,
    branch: str = "main",
    metadata: dict | None = None,
) -> str:
    """Commit all staged changes on *branch* and return the commit ID."""
    client = lakefs.Client(host=url, username=user, password=password)
    ref = lakefs.repository(repo, client=client).branch(branch).commit(
        message=message,
        metadata=metadata or {},
    )
    return ref.id
