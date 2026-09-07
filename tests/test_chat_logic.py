from host.chat import _content_text, _last_message_text, _map_finish_reason, to_sdk_mode


def test_to_sdk_mode():
    assert to_sdk_mode("ask") == "plan"
    assert to_sdk_mode("plan") == "plan"
    assert to_sdk_mode("agent") == "agent"


def test_map_finish_reason():
    assert _map_finish_reason("finished") == "stop"
    assert _map_finish_reason("cancelled") == "cancel"
    assert _map_finish_reason("error") == "error"
    assert _map_finish_reason("expired") == "error"
    assert _map_finish_reason("unknown") == "stop"


def test_content_text_string():
    assert _content_text("hello") == "hello"
    assert _content_text(None) == ""


def test_content_text_list_of_parts():
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert _content_text(parts) == "a b"


def test_last_message_text():
    msgs = [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]
    assert _last_message_text(msgs) == "second"
    assert _last_message_text([]) == ""
