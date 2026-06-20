from topic_overviews.state import State, load_state, save_state


def test_load_missing_returns_empty(tmp_path):
    st = load_state(str(tmp_path / "nope.json"))
    assert st.last_harvest is None
    assert st.seen_ids == set()


def test_save_then_load_roundtrip(tmp_path):
    path = str(tmp_path / "state.json")
    save_state(path, State(last_harvest="2026-06-19", seen_ids={"2401.00001", "2401.00002"}))
    st = load_state(path)
    assert st.last_harvest == "2026-06-19"
    assert st.seen_ids == {"2401.00001", "2401.00002"}
