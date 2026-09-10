"""Regression coverage for #7481 — serial custom-provider probes starving.

During a cold model-catalog rebuild the WebUI probes the active endpoint first
(``model.base_url``) and then each named ``custom_providers`` entry, serially,
off one shared ``_LIVE_REBUILD_BUDGET_SECONDS`` budget while each probe is
individually capped at ``CUSTOM_MODELS_ENDPOINT_TIMEOUT_SECONDS``.

Before the fix one unreachable endpoint — the common case being a LAN
LM Studio/Ollama host the webui container cannot route to, because the probe runs
server-side — spent the whole budget on its own connect timeout, so every
reachable provider scheduled behind it never got an in-band ``/v1/models``
probe: its group rendered from a stale disk cache or stayed empty, and every
cold load paid the full stall.

The fix (``_CustomProbeSchedule``) gives every probe a fair slice of the
remaining window instead of the whole cap, and the LM Studio provider-group
fallback — a second consumer of the same dead endpoint, previously on a
hardcoded 5s timeout outside any budget — now draws from the same schedule.
"""

from __future__ import annotations

import copy
import json
import socket
import time
import urllib.error
import urllib.request

import pytest

import api.config as cfg
import api.profiles as profiles


# Models the reachable gateway advertises — what the picker must show in-band.
_GATEWAY_MODELS = ["gateway-model-a", "gateway-model-b"]

# The issue's own numbers, scaled down only where noted.
_BUDGET = 4.0
_CAP = 1.5


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body


def _install_urlopen(monkeypatch, *, dead_hosts, live_hosts):
    """Route probes by host and record the timeout each one was handed.

    A "dead" host behaves the way an unreachable LAN endpoint does: it burns the
    entire timeout it was given and then fails — exactly the behaviour that used
    to eat the shared rebuild budget.

    Returns ``{"dead": [(url, timeout)], "live": [(url, timeout)]}``.
    """
    observed: dict[str, list] = {"dead": [], "live": []}

    def fake_urlopen(req, timeout=None):
        url = str(getattr(req, "full_url", ""))
        if any(host in url for host in dead_hosts):
            observed["dead"].append((url, timeout))
            time.sleep(timeout if timeout is not None else 10)
            raise urllib.error.URLError("timed out")
        if any(host in url for host in live_hosts):
            observed["live"].append((url, timeout))
            return _FakeResponse({"data": [{"id": mid} for mid in _GATEWAY_MODELS]})
        raise urllib.error.URLError(f"unexpected probe: {url}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return observed


@pytest.fixture(autouse=True)
def isolate_models_catalog_state(monkeypatch, tmp_path):
    """Hermetic catalog state, mirroring the #3928 budget-fallback fixture."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")
    auth_store_path = tmp_path / "auth.json"
    auth_store_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cfg, "_get_config_path", lambda: config_path)
    monkeypatch.setattr(cfg, "_cfg_path", config_path, raising=False)
    monkeypatch.setattr(cfg, "_cfg_mtime", config_path.stat().st_mtime, raising=False)
    monkeypatch.setattr(cfg, "_cfg_has_in_memory_overrides", lambda: True)
    monkeypatch.setattr(cfg, "_get_auth_store_path", lambda: auth_store_path)
    monkeypatch.setattr(cfg, "_load_models_cache_from_disk", lambda: None)
    monkeypatch.setattr(cfg, "_save_models_cache_to_disk", lambda *_a, **_k: None)
    monkeypatch.setattr(cfg, "_get_models_cache_path", lambda: tmp_path / "models_cache.json")
    monkeypatch.setattr(cfg, "_delete_models_cache_on_disk", lambda: None)
    monkeypatch.setattr(cfg, "_models_cache_source_fingerprint", lambda: "issue-7481-fp")
    monkeypatch.setattr(cfg, "_available_models_cache", None, raising=False)
    monkeypatch.setattr(cfg, "_available_models_cache_ts", 0.0, raising=False)
    monkeypatch.setattr(cfg, "_available_models_live_rebuild_ts", 0.0, raising=False)
    monkeypatch.setattr(cfg, "_available_models_cache_source_fingerprint", None, raising=False)
    monkeypatch.setattr(cfg, "_cache_build_in_progress", False, raising=False)
    monkeypatch.setattr(cfg, "_models_rebuild_seq", 0, raising=False)
    monkeypatch.setattr(cfg, "_models_published_seq", 0, raising=False)
    monkeypatch.setattr(cfg, "cfg", {}, raising=False)
    # Any provider left in the catalog would otherwise shell out to the Hermes
    # CLI for a live id list; the rebuild must stay network-free apart from the
    # custom endpoints under test.
    monkeypatch.setattr(cfg, "_read_live_provider_model_ids", lambda _pid: [])
    monkeypatch.setattr(profiles, "get_active_hermes_home", lambda: tmp_path / "hermes-home")
    monkeypatch.setattr(cfg.os, "getenv", lambda key, default=None: default or "")
    # The probe path resolves the endpoint hostname for its SSRF guard. A real
    # resolver makes these tests depend on the host's DNS behaviour (and this
    # container takes seconds to answer NXDOMAIN), so pin it to an immediate
    # failure: the guard treats that as "not resolvable" and lets the probe
    # through, which is exactly what the fake urlopen above is standing in for.
    def _unresolvable(host, port, *args, **kwargs):
        raise socket.gaierror("hermetic test resolver")

    monkeypatch.setattr(socket, "getaddrinfo", _unresolvable)

    return {"tmp_path": tmp_path, "auth_store_path": auth_store_path}


def _configure(monkeypatch, *, active_base_url, provider_base_url=None, custom_providers=None):
    cfg.cfg = {
        "model": {
            "provider": "lmstudio",
            "default": "some-local-model",
            "base_url": active_base_url,
        },
        "providers": (
            {"lmstudio": {"base_url": provider_base_url}} if provider_base_url else {}
        ),
        "fallback_providers": [],
        "custom_providers": custom_providers or [],
    }
    monkeypatch.setattr(cfg, "_LIVE_REBUILD_BUDGET_SECONDS", _BUDGET, raising=False)
    monkeypatch.setattr(cfg, "CUSTOM_MODELS_ENDPOINT_TIMEOUT_SECONDS", _CAP, raising=False)


def _bare_id(model_id: str) -> str:
    """``@custom:my-gateway:model-a`` -> ``model-a``."""
    return str(model_id).split(":")[-1]


def _models_by_provider(catalog: dict) -> dict[str, list[str]]:
    return {
        group["provider_id"]: [_bare_id(m.get("id")) for m in group.get("models", [])]
        for group in catalog["groups"]
    }


class _FakeClock:
    """Stand-in for the ``time`` module so a schedule can be advanced by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


def _catalog(name: str) -> dict:
    """A minimal catalog tagged so a test can tell which build produced it."""
    return {
        "active_provider": name,
        "default_model": f"{name}/model",
        "configured_model_badges": {},
        "groups": [{"provider": name.title(), "provider_id": name, "models": []}],
        "aliases": {},
    }


def test_unreachable_lan_active_endpoint_does_not_starve_the_gateway_behind_it(
    monkeypatch, isolate_models_catalog_state
):
    """The #7481 repro, using the issue's own config shape.

    A dead LAN endpoint is configured both as the active provider and as
    ``providers.lmstudio`` (so the provider-group fallback probes it a second
    time). The reachable named gateway behind it must still get its in-band
    probe and land in the catalog the caller receives — not merely be deferred
    to an out-of-band refresh after a fallback was served.
    """
    _configure(
        monkeypatch,
        active_base_url="http://lan-dead.example:1234/v1",
        provider_base_url="http://lan-dead.example:1234/v1",
        custom_providers=[
            {
                "name": "My Gateway",
                "base_url": "https://gw-live.example/v1",
                "api_key": "sk-live",
            }
        ],
    )
    observed = _install_urlopen(
        monkeypatch, dead_hosts=["lan-dead.example"], live_hosts=["gw-live.example"]
    )

    started = time.monotonic()
    catalog = cfg.get_available_models()
    elapsed = time.monotonic() - started

    # The rebuild published in-band: no budget-exceeded fallback was served.
    assert elapsed < _BUDGET
    assert observed["live"], "the reachable named provider was never probed"
    assert _models_by_provider(catalog).get("custom:my-gateway") == _GATEWAY_MODELS

    # Both consumers of the dead endpoint were attempted — the active-endpoint
    # probe and the LM Studio provider-group fallback — and each was handed a
    # bounded slice rather than the whole window.
    assert len(observed["dead"]) == 2
    for _url, timeout in observed["dead"]:
        assert timeout is not None and timeout < _CAP


def test_every_dead_endpoint_in_the_chain_still_lets_the_live_one_through(
    monkeypatch, isolate_models_catalog_state
):
    """Two dead named providers in front of a reachable one must not starve it."""
    _configure(
        monkeypatch,
        active_base_url="http://lan-dead.example:1234/v1",
        custom_providers=[
            {"name": "Dead One", "base_url": "https://dead-one.example/v1", "api_key": "k1"},
            {"name": "Dead Two", "base_url": "https://dead-two.example/v1", "api_key": "k2"},
            {"name": "My Gateway", "base_url": "https://gw-live.example/v1", "api_key": "k3"},
        ],
    )
    observed = _install_urlopen(
        monkeypatch,
        dead_hosts=["lan-dead.example", "dead-one.example", "dead-two.example"],
        live_hosts=["gw-live.example"],
    )

    catalog = cfg.get_available_models()

    # Every named endpoint was attempted, in config order, before the budget
    # ran out (the active endpoint may additionally be re-probed by the LM
    # Studio provider-group fallback, which is the same window).
    probed_hosts = [url.split("://", 1)[1].split("/", 1)[0] for url, _ in observed["dead"]]
    assert [host for host in probed_hosts if host != "lan-dead.example:1234"] == [
        "dead-one.example",
        "dead-two.example",
    ]
    assert _models_by_provider(catalog).get("custom:my-gateway") == _GATEWAY_MODELS


def test_static_allowlist_provider_is_never_probed_and_does_not_dilute_the_schedule(
    monkeypatch, isolate_models_catalog_state
):
    """A provider with a static ``models:`` allowlist consumes no probe slot."""
    _configure(
        monkeypatch,
        active_base_url="http://lan-dead.example:1234/v1",
        custom_providers=[
            {
                "name": "Static Co",
                "base_url": "https://static-never-probed.example/v1",
                "api_key": "k",
                "models": ["static-a", "static-b"],
            },
            {"name": "My Gateway", "base_url": "https://gw-live.example/v1", "api_key": "k"},
        ],
    )
    observed = _install_urlopen(
        monkeypatch,
        dead_hosts=["lan-dead.example"],
        live_hosts=["gw-live.example", "static-never-probed.example"],
    )

    catalog = cfg.get_available_models()

    # The allowlist provider rendered from config and was never probed…
    assert _models_by_provider(catalog).get("custom:static-co") == ["static-a", "static-b"]
    assert not [
        url
        for url, _ in observed["live"] + observed["dead"]
        if "static-never-probed.example" in url
    ]
    # …while the live provider behind the dead endpoint still made it in-band.
    assert _models_by_provider(catalog).get("custom:my-gateway") == _GATEWAY_MODELS


def test_probe_schedule_keeps_the_documented_cap_when_the_budget_is_disabled(monkeypatch):
    """Legacy synchronous path (budget <= 0) has no window to share."""
    monkeypatch.setattr(cfg, "_LIVE_REBUILD_BUDGET_SECONDS", 0.0, raising=False)
    monkeypatch.setattr(cfg, "CUSTOM_MODELS_ENDPOINT_TIMEOUT_SECONDS", 5.0, raising=False)

    schedule = cfg._CustomProbeSchedule(3)

    assert [schedule.next_timeout() for _ in range(3)] == [5.0, 5.0, 5.0]


def test_probe_schedule_shares_the_window_and_reserves_headroom(monkeypatch):
    """Every probe gets a slice, and a fully-burned chain cannot drain the window."""
    monkeypatch.setattr(cfg, "_LIVE_REBUILD_BUDGET_SECONDS", 4.0, raising=False)
    monkeypatch.setattr(cfg, "CUSTOM_MODELS_ENDPOINT_TIMEOUT_SECONDS", 5.0, raising=False)

    clock = _FakeClock()
    monkeypatch.setattr(cfg, "time", clock, raising=False)

    schedule = cfg._CustomProbeSchedule(4)
    timeouts = []
    for _ in range(4):
        timeout = schedule.next_timeout()
        timeouts.append(timeout)
        clock.now += timeout  # the probe burns its whole slice

    assert all(t < 5.0 for t in timeouts), timeouts
    # A fully-burned chain still finishes inside the window, so the foreground
    # caller gets a published catalog instead of the over-budget fallback.
    assert clock.now < 4.0, timeouts


def test_probe_schedule_cannot_outspend_the_window_at_any_chain_length(monkeypatch):
    """The headroom must survive long chains — `custom_providers` is unbounded.

    Regression guard for the review finding on the first revision: a fixed 0.5s
    per-probe floor let eight timeouts spend the whole four-second window and
    nine spend past it, so a long chain of dead endpoints could still push a
    reachable provider out of the in-band rebuild.
    """
    monkeypatch.setattr(cfg, "_LIVE_REBUILD_BUDGET_SECONDS", 4.0, raising=False)
    monkeypatch.setattr(cfg, "CUSTOM_MODELS_ENDPOINT_TIMEOUT_SECONDS", 5.0, raising=False)

    clock = _FakeClock()
    monkeypatch.setattr(cfg, "time", clock, raising=False)

    for endpoint_count in (1, 2, 8, 24):
        clock.now = 0.0
        schedule = cfg._CustomProbeSchedule(endpoint_count)
        for _ in range(endpoint_count):
            clock.now += schedule.next_timeout()  # every probe burns its slice
        assert clock.now < 4.0, (endpoint_count, clock.now)


def test_probe_schedule_restores_the_cap_once_the_budget_is_spent(monkeypatch):
    """The out-of-band continuation still gets a full attempt to refresh."""
    monkeypatch.setattr(cfg, "_LIVE_REBUILD_BUDGET_SECONDS", 0.05, raising=False)
    monkeypatch.setattr(cfg, "CUSTOM_MODELS_ENDPOINT_TIMEOUT_SECONDS", 5.0, raising=False)

    schedule = cfg._CustomProbeSchedule(3)
    assert schedule.next_timeout() < 5.0
    time.sleep(0.1)  # budget now spent
    assert schedule.next_timeout() == 5.0


def test_late_out_of_band_result_cannot_overwrite_a_newer_rebuild(
    monkeypatch, isolate_models_catalog_state
):
    """A superseded rebuild must not resurrect its catalog over a newer one."""
    _configure(monkeypatch, active_base_url=None)
    monkeypatch.setattr(cfg, "_LIVE_REBUILD_BUDGET_SECONDS", 0.05, raising=False)

    newer = _catalog("newer")
    older = _catalog("older")
    finished = {"value": False}

    def _slow_builder(_builder):
        # Still running when the foreground gives up — and by now a NEWER
        # rebuild has been allocated and published its catalog (the out-of-band
        # race the guard exists for).
        time.sleep(0.15)
        cfg._models_rebuild_seq += 1
        cfg._models_published_seq = cfg._models_rebuild_seq
        cfg._available_models_cache = newer
        cfg._available_models_cache_ts = time.monotonic()
        cfg._available_models_live_rebuild_ts = time.monotonic()
        finished["value"] = True
        return copy.deepcopy(older)

    monkeypatch.setattr(cfg, "_invoke_models_rebuild", _slow_builder)

    cfg.get_available_models()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not finished["value"]:
        time.sleep(0.01)
    # Give the worker's finally-block publisher a moment to (not) clobber.
    time.sleep(0.15)

    assert finished["value"] is True
    assert cfg._available_models_cache is newer, (
        "the superseded out-of-band rebuild overwrote a newer catalog"
    )


def test_older_publish_does_not_cost_a_newer_rebuild_its_result(
    monkeypatch, isolate_models_catalog_state
):
    """An older build publishing late must not suppress the newer build's publish.

    Regression guard for the review finding on the first revision: ordering by
    wall-clock stamp meant an OLDER rebuild that published *after* a newer one
    had started read as the newer generation, so the newer build's correct result
    was discarded — leaving a stale catalog in the cache and disk while its
    caller received a catalog that was never published.
    """
    _configure(monkeypatch, active_base_url=None)
    monkeypatch.setattr(cfg, "_LIVE_REBUILD_BUDGET_SECONDS", 5.0, raising=False)

    newer = _catalog("newer")
    older = _catalog("older")

    def _builder(_builder):
        # This run is rebuild N; simulate rebuild N-1 publishing its older
        # catalog late, while we are still building.
        cfg._models_published_seq = cfg._models_rebuild_seq - 1
        cfg._available_models_cache = older
        cfg._available_models_cache_ts = time.monotonic()
        cfg._available_models_live_rebuild_ts = time.monotonic()
        return copy.deepcopy(newer)

    monkeypatch.setattr(cfg, "_invoke_models_rebuild", _builder)

    result = cfg.get_available_models()

    assert result["active_provider"] == "newer"
    assert cfg._available_models_cache is not older, (
        "an older publish suppressed the newer rebuild's result"
    )
    assert cfg._available_models_cache["active_provider"] == "newer"
    assert cfg._models_published_seq == cfg._models_rebuild_seq
