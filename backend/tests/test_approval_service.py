from app.services.approval_service import canonical_payload_hash, hash_token


def test_canonical_payload_hash_is_order_independent() -> None:
    payload_a = {"target": "host1", "cmd": "reboot", "args": {"force": True, "delay": 5}}
    payload_b = {"args": {"delay": 5, "force": True}, "cmd": "reboot", "target": "host1"}
    assert canonical_payload_hash(payload_a) == canonical_payload_hash(payload_b)


def test_hash_token_is_deterministic() -> None:
    token = "abc123-token-value"
    assert hash_token(token) == hash_token(token)
    assert hash_token(token) != hash_token("different-token")

