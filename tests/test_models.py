from host.models import ModelsCache, _expand_with_modes


class FakeRuntime:
    client = None


def make_cache():
    return ModelsCache(FakeRuntime())


def test_base_model_id_splits_known_suffix():
    cache = make_cache()
    assert cache.base_model_id("composer-2.5:agent") == ("composer-2.5", "agent")
    assert cache.base_model_id("composer-2.5:plan") == ("composer-2.5", "plan")
    assert cache.base_model_id("composer-2.5") == ("composer-2.5", None)


def test_base_model_id_ignores_unknown_suffix():
    cache = make_cache()
    assert cache.base_model_id("some:custom") == ("some:custom", None)


def test_context_length_uses_map_and_default():
    cache = make_cache()
    cache._context_limits = {"default": 111, "composer-2.5": 222}
    assert cache.context_length("composer-2.5") == 222
    assert cache.context_length("composer-2.5:agent") == 222
    assert cache.context_length("unknown-model") == 111


def test_expand_with_modes_adds_suffixed_variants():
    base = [{"id": "m1", "object": "model"}]
    out = _expand_with_modes(base, 0)
    ids = {m["id"] for m in out}
    assert {"m1", "m1:ask", "m1:plan", "m1:agent"} <= ids
