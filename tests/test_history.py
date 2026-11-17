import os
import pytest
from app.core.history_manager import HistoryManager

@pytest.fixture
def temp_history(tmp_path):
    db_file = str(tmp_path / "test_history.db")
    mgr = HistoryManager(db_path=db_file)
    return mgr

def test_add_and_list_records(temp_history):
    temp_history.add_record(
        record_id="rec_001",
        engine="edge",
        voice_id="yunxi",
        voice_name="云希",
        text="测试文本",
        audio_url="/static/generated/test.mp3",
        duration=1.5,
        cost_sec=0.2,
        rtf=0.133,
        sample_rate=24000
    )
    records = temp_history.list_records(limit=10)
    assert len(records) == 1
    assert records[0]["id"] == "rec_001"
    assert records[0]["voice_name"] == "云希"

def test_toggle_favorite(temp_history):
    temp_history.add_record(
        record_id="rec_fav",
        engine="edge",
        voice_id="yunxi",
        voice_name="云希",
        text="收藏测试",
        audio_url="/static/generated/test.mp3",
        duration=1.0,
        cost_sec=0.1,
        rtf=0.1,
        sample_rate=24000
    )
    temp_history.update_tag_and_note("rec_fav", is_favorite=True)
    records = temp_history.list_records(favorite_only=True)
    assert len(records) == 1
    assert records[0]["id"] == "rec_fav"
    assert records[0]["is_favorite"] is True

    temp_history.update_tag_and_note("rec_fav", is_favorite=False)
    records_after = temp_history.list_records(favorite_only=True)
    assert len(records_after) == 0

def test_dialogue_turns(temp_history):
    temp_history.add_dialogue_turn(
        turn_id="turn_01",
        user_text="你好",
        assistant_reply="你好呀！",
        audio_url="/static/generated/turn_01.wav",
        engine="edge",
        voice_id="yunxi",
        duration=1.2,
        cost_sec=0.3,
        rtf=0.25,
        mode="text"
    )
    dialogues = temp_history.list_dialogue_history(limit=5)
    assert len(dialogues) == 1
    assert dialogues[0]["user_text"] == "你好"
    assert dialogues[0]["assistant_reply"] == "你好呀！"
