from topic_overviews.config import load_config


def test_load_config_reads_env_and_defaults():
    env = {
        "ANTHROPIC_API_KEY": "sk-test",
        "TOPIC_OVERVIEWS_OVERVIEW_TOPIC_QID": "Q123",
        "TOPIC_OVERVIEWS_DRY_RUN": "true",
    }
    cfg = load_config(env)
    assert cfg.anthropic_api_key == "sk-test"
    assert cfg.overview_topic_qid == "Q123"
    assert cfg.dry_run is True
    assert cfg.model == "claude-haiku-4-5"        # default
    assert cfg.arxiv_set == "math"                # default
    assert cfg.relevance_threshold == 0.0         # default, coerced to float


def test_dry_run_defaults_false():
    cfg = load_config({})
    assert cfg.dry_run is False
