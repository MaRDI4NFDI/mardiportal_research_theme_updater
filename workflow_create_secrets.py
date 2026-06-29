"""One-time setup: create Prefect Secret blocks for the topic-overviews flow.

Run once against the MaRDI Prefect server:

    PREFECT_API_URL=http://prefect-mardi.zib.de/api \
    PREFECT_API_AUTH_STRING="admin:<password>" \
    python workflow_create_secrets.py

You will be prompted for each secret value interactively.
Re-running overwrites existing blocks (overwrite=True).
"""
import getpass
from prefect.blocks.system import Secret


def create(name: str, prompt: str, *, secret: bool = True) -> None:
    value = getpass.getpass(f"{prompt}: ") if secret else input(f"{prompt}: ")
    Secret(value=value).save(name, overwrite=True)
    print(f"  ✓  {name}")


if __name__ == "__main__":
    print("Creating Prefect Secret blocks for topic-overviews\n")
    create("topic-overviews-bot-user",        "MediaWiki bot username (e.g. DoipBot)", secret=False)
    create("topic-overviews-bot-password",    "MediaWiki bot password")
    create("topic-overviews-openai-api-key",  "ZIB Ollama API key (OLLAMA_API_KEY)")
    create("topic-overviews-s2-api-key",      "Semantic Scholar API key (S2_API_KEY)")
    # lakeFS credentials — sourced from k8s secret in production namespace.
    # Confirm secret name/keys first: kubectl get secret -n production lakefs -o jsonpath='{.data.access-key}' | base64 -d
    create("topic-overviews-lakefs-user",     "lakeFS access key ID")
    create("topic-overviews-lakefs-password", "lakeFS secret access key")
    print("\nDone.")
