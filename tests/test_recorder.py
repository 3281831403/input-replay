"""Acceptance tests for the deterministic input replay pipeline.

The tests are intentionally read-only for the candidate: every contract they
exercise is documented in ``replay/recorder.py``.  No test uses randomness,
sleeps, touches the network, or depends on wall-clock time.
"""

import pytest

from replay.recorder import (
    MAX_EVENTS,
    ReplayError,
    align_to_frames,
    apply_deadzone,
    checksum,
    dedupe,
    merge_tracks,
    record,
    replay,
    switch_device,
)


def test_record_uses_variable_frame_times():
    """A 0/40/110 ms frame grid must not be treated as fixed 60 Hz."""
    frame_times = [0, 40, 110]
    events = [
        {"t": 5, "type": "press", "payload": {"key": "A"}},
        {"t": 39, "type": "move", "payload": {"x": 1}},
        {"t": 41, "type": "move", "payload": {"x": 2}},
        {"t": 109, "type": "move", "payload": {"x": 3}},
        {"t": 110, "type": "release", "payload": {"key": "A"}},
    ]

    track = record(events, frame_times)

    assert [event["frame"] for event in track["events"]] == [0, 0, 1, 1, 2]
    assert track["frame_times"] == frame_times
    assert track["truncated"] is False


def test_align_uses_nearest_frame_at_boundary():
    """Snapping uses distance, not the interval before the boundary."""
    frame_times = [0, 100, 200]
    events = [
        {"t": 99, "type": "move", "payload": {"x": 1}},
        {"t": 100, "type": "press", "payload": {"key": "A"}},
        {"t": 151, "type": "move", "payload": {"x": 2}},
    ]

    aligned = align_to_frames(events, frame_times, tolerance=50)

    assert [event["frame"] for event in aligned] == [1, 1, 2]
    assert [event["delta_ms"] for event in aligned] == [-1, 0, -49]


def test_dedupe_keeps_press_release_pair():
    """Only the repeated press collapses; the press/release pair survives."""
    events = [
        {"t": 10, "type": "press", "payload": {"key": "A"}, "frame": 2},
        {"t": 11, "type": "press", "payload": {"key": "A"}, "frame": 2},
        {"t": 12, "type": "release", "payload": {"key": "A"}, "frame": 2},
    ]

    deduped = dedupe(events)

    assert [(event["type"], event["t"]) for event in deduped] == [
        ("press", 11),
        ("release", 12),
    ]
    assert all(event["payload"]["key"] == "A" for event in deduped)


def test_deadzone_full_deflection_reaches_sensitivity():
    """The normalized formula must recover the full output range."""
    assert apply_deadzone(1.0, 0.2, 1.0) == pytest.approx(1.0)
    assert apply_deadzone(-1.0, 0.2, 1.0) == pytest.approx(-1.0)
    assert apply_deadzone(0.6, 0.2, 1.0) == pytest.approx(0.5)


def test_switch_device_inserts_boundary_event():
    """Changing devices is an explicit boundary event, not just metadata."""
    track = {
        "frame_times": [0, 16, 32],
        "events": [
            {"t": 5, "type": "press", "payload": {"key": "A"}, "frame": 0},
        ],
        "device": "keyboard",
    }

    switched = switch_device(track, "wheel")

    assert track["device"] == "keyboard"
    assert len(track["events"]) == 1
    assert switched["device"] == "wheel"
    boundaries = [
        event for event in switched["events"] if event["type"] == "device_switch"
    ]
    assert len(boundaries) == 1
    boundary = boundaries[0]
    assert boundary["payload"]["device_id"] == "wheel"
    assert boundary["payload"]["from"] == "keyboard"
    assert boundary["t"] > 5
    assert boundary["frame"] == 0
    assert switched["events"][-1] == boundary


def test_checksum_ignores_key_and_event_order():
    """Equal content hashes equally even when dicts/events are reordered."""
    track_a = {
        "frame_times": [0, 16],
        "device": "pad",
        "device_priority": 2,
        "events": [
            {"t": 5, "type": "move", "payload": {"x": 2, "y": 3}, "frame": 0},
            {"t": 7, "type": "press", "payload": {"key": "A"}, "frame": 0},
        ],
    }
    track_b = {
        "device_priority": 2,
        "device": "pad",
        "events": [
            {"frame": 0, "payload": {"key": "A"}, "type": "press", "t": 7},
            {"frame": 0, "payload": {"y": 3, "x": 2}, "type": "move", "t": 5},
        ],
        "frame_times": [0, 16],
    }
    track_c = {
        "frame_times": [0, 16],
        "device": "pad",
        "device_priority": 2,
        "events": [
            {"t": 6, "type": "move", "payload": {"x": 2, "y": 3}, "frame": 0},
            {"t": 7, "type": "press", "payload": {"key": "A"}, "frame": 0},
        ],
    }

    assert checksum(track_a) == checksum(track_b)
    assert checksum(track_a) != checksum(track_c)


def test_replay_delivers_each_event_once():
    """Two events in one frame do not cause duplicate callbacks."""
    frame_times = [0, 16]
    track = {
        "frame_times": frame_times,
        "device": "pad",
        "events": [
            {"t": 1, "type": "press", "payload": {"key": "A"}, "frame": 0},
            {"t": 2, "type": "release", "payload": {"key": "A"}, "frame": 0},
            {"t": 17, "type": "press", "payload": {"key": "B"}, "frame": 1},
        ],
    }
    seen = []

    def handler(event):
        seen.append(
            (event["frame"], event["t"], event["type"], event["payload"].get("key"))
        )

    count = replay(track, frame_times, handler)

    assert count == 3
    assert seen == [
        (0, 1, "press", "A"),
        (0, 2, "release", "A"),
        (1, 17, "press", "B"),
    ]
    assert len(seen) == len(set(seen))


def test_merge_prefers_high_priority_device():
    """A later, lower-priority track must not replace a higher-priority event."""
    high_priority = {
        "frame_times": [0, 16],
        "device": "wheel",
        "device_priority": 10,
        "events": [
            {
                "t": 5,
                "type": "press",
                "payload": {"key": "fire", "source": "wheel"},
                "frame": 0,
            },
        ],
    }
    low_priority = {
        "frame_times": [0, 16],
        "device": "pad",
        "device_priority": 1,
        "events": [
            {
                "t": 6,
                "type": "press",
                "payload": {"key": "fire", "source": "pad"},
                "frame": 0,
            },
            {"t": 18, "type": "press", "payload": {"key": "jump"}, "frame": 1},
        ],
    }

    merged = merge_tracks([high_priority, low_priority])

    fire = [
        event for event in merged["events"] if event["payload"].get("key") == "fire"
    ]
    assert len(fire) == 1
    assert fire[0]["payload"]["source"] == "wheel"
    assert any(
        event["payload"].get("key") == "jump" for event in merged["events"]
    )


def test_record_truncates_oversized_recording():
    """Overlong input is capped so memory cannot grow without bound."""
    events = [
        {"t": index, "type": "tick", "payload": {"seq": index}}
        for index in range(MAX_EVENTS + 1)
    ]

    track = record(events, [0])

    assert track["truncated"] is True
    assert len(track["events"]) == MAX_EVENTS
    assert track["events"][0]["payload"]["seq"] == 0
    assert track["events"][-1]["payload"]["seq"] == MAX_EVENTS - 1


def test_record_rejects_timestamp_regression():
    """A backwards timestamp is an error, not a silently reordered replay."""
    events = [
        {"t": 100, "type": "press", "payload": {"key": "A"}},
        {"t": 99, "type": "release", "payload": {"key": "A"}},
    ]

    with pytest.raises(ReplayError):
        record(events, [0, 100])


def test_checksum_stable_for_same_track():
    """Hashing the same track twice is deterministic."""
    track = {
        "frame_times": [0, 16],
        "events": [
            {"t": 3, "type": "press", "payload": {"key": "A"}, "frame": 0},
        ],
    }

    assert checksum(track) == checksum(track)


def test_replay_empty_track_returns_zero():
    """No events means no callbacks and a zero event count."""
    calls = []

    def handler(event):
        calls.append(event)

    count = replay({"frame_times": [0, 16], "events": []}, [0, 16], handler)

    assert count == 0
    assert calls == []


def test_deadzone_zero_is_identity():
    """A zero dead zone with unit sensitivity leaves the axis untouched."""
    assert apply_deadzone(0.37, 0.0, 1.0) == pytest.approx(0.37)
    assert apply_deadzone(-0.37, 0.0, 1.0) == pytest.approx(-0.37)


def test_deadzone_at_least_one_raises():
    """A dead zone of 1 or more makes normalization impossible."""
    for deadzone in (1.0, 1.5):
        with pytest.raises(ReplayError):
            apply_deadzone(0.5, deadzone, 1.0)


def test_merge_frame_count_is_union():
    """Merging keeps every frame from every input, including empty frames."""
    wide = {
        "frame_times": [0, 16, 33, 50],
        "device": "wheel",
        "device_priority": 2,
        "events": [
            {"t": 1, "type": "press", "payload": {"key": "a"}, "frame": 0},
            {"t": 35, "type": "press", "payload": {"key": "c"}, "frame": 2},
        ],
    }
    narrow = {
        "frame_times": [0, 16, 33],
        "device": "pad",
        "device_priority": 1,
        "events": [
            {"t": 17, "type": "press", "payload": {"key": "b"}, "frame": 1},
        ],
    }

    merged = merge_tracks([wide, narrow])

    assert merged["frame_times"] == [0, 16, 33, 50]
    assert len(merged["frame_times"]) == len({0, 16, 33, 50})
    assert {event["payload"]["key"] for event in merged["events"]} == {"a", "b", "c"}
