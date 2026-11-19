"""Deterministic replay primitives for recorded input events.

All timestamps are integer milliseconds.  Frames are described by a strictly
increasing list of frame start times, so a recording may use a variable frame
rate.  Events have the shape::

    {"t": int, "type": str, "payload": dict}

The functions in this module are deliberately left as stubs: every public
entry point raises ``NotImplementedError`` until the replay pipeline is fixed.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left, bisect_right

MAX_EVENTS = 10_000
DEVICE_SWITCH = "device_switch"


class ReplayError(Exception):
    """Raised when a recording does not satisfy the replay contract."""


def record(events, frame_times):
    """Archive events into frame numbers using the real, variable frame spacing.

    Time unit: milliseconds.  ``frame_times`` is a strictly increasing list of
    frame start times and must contain at least one frame.  Each input event is
    a mapping with at least ``{"t": int, "type": str, "payload": dict}``, and
    the input order is the order in which the events happened.  A timestamp
    that goes backwards must raise :class:`ReplayError`; it must never be
    silently reordered.

    An event belongs to the last frame whose start time is ``<= event["t"]``.
    Events before the first frame belong to frame 0, and events at or after the
    last frame belong to the last frame.  A fixed 1/60 s step is not valid here:
    only the supplied ``frame_times`` may be used.

    Returns a new track dict::

        {
            "frame_times": list[int],   # copy of the input
            "events": list[dict],       # event copies plus "frame": int
            "truncated": bool,
        }

    At most ``MAX_EVENTS`` events are retained.  Longer recordings keep the
    first ``MAX_EVENTS`` events and set ``truncated`` to ``True`` so memory is
    bounded.  The input objects are not mutated.
    """
    raise NotImplementedError("record() is not implemented")


def align_to_frames(events, frame_times, tolerance):
    """Snap events to the nearest frame, dropping events that are too far away.

    Time unit: milliseconds; ``tolerance`` is also in milliseconds and must be
    non-negative.  ``frame_times`` must be strictly increasing.  For every
    event, the nearest frame start is chosen.  An exact tie between two frames
    goes to the later frame.  If ``abs(event["t"] - frame_time) > tolerance``
    the event is dropped.

    Returns a new list in input order.  Every retained event is a copy of the
    input event with two added fields::

        "frame": int      # index into frame_times
        "delta_ms": int   # event["t"] - frame_times[frame]

    The input events are not modified.  Invalid frame lists or tolerances raise
    :class:`ReplayError`.
    """
    raise NotImplementedError("align_to_frames() is not implemented")


def dedupe(events):
    """Collapse duplicate same-frame key events, keeping the last occurrence.

    Events must already carry an integer ``frame`` field.  Two events are the
    same logical duplicate when their ``frame``, ``type`` and
    ``payload["key"]`` all match.  For each such group only the last event is
    kept, in the order of those retained occurrences.  Events without a
    ``payload["key"]`` are never deduplicated.

    A press and a release of the same key are different event types and must
    both survive, even when they occur in the same frame; merging them would
    break the replay duration.  The input list is not modified.  Missing
    ``frame``/``type`` fields raise :class:`ReplayError`.
    """
    raise NotImplementedError("dedupe() is not implemented")


def apply_deadzone(axis, deadzone, sensitivity):
    """Apply a symmetric dead zone and normalize the remaining travel.

    ``axis`` is a finite value in ``[-1.0, 1.0]``.  ``deadzone`` is in
    ``[0.0, 1.0)`` and ``sensitivity`` is non-negative.  Inside the dead zone
    (``abs(axis) <= deadzone``) the result is exactly ``0.0``.  Outside it the
    normalized magnitude is::

        (abs(axis) - deadzone) / (1.0 - deadzone) * sensitivity

    with the sign of ``axis`` preserved.  Dividing by ``1 - deadzone`` is
    required: at full deflection the result must reach ``sensitivity`` (for
    example ``apply_deadzone(1.0, 0.2, 1.0) == 1.0``), not merely
    ``1.0 - deadzone``.

    Returns a float.  A deadzone ``>= 1`` (or negative), a negative
    sensitivity, a non-finite input, or an out-of-range axis raises
    :class:`ReplayError`.
    """
    raise NotImplementedError("apply_deadzone() is not implemented")


def switch_device(track, device_id):
    """Return a copy of ``track`` with a device-switch boundary appended.

    The boundary is an ordinary event::

        {
            "t": int,                    # milliseconds
            "type": "device_switch",
            "payload": {"device_id": device_id, "from": previous_device},
            "frame": int,
        }

    It is placed after the existing event stream: when events exist its time is
    the last event's ``t`` plus one millisecond and it uses that last event's
    frame; for an event-less track it uses the last frame time/index, or 0/0
    when the track has no frames either.  The returned track's ``"device"`` is
    ``device_id``.  The original track is not mutated.

    Replay uses this boundary to segment the stream: events before it use the
    old device state and events after it use the new device state.
    """
    raise NotImplementedError("switch_device() is not implemented")


def checksum(track):
    """Return a stable SHA-256 hex digest for a track's content.

    Mapping keys are sorted recursively before serialization, and the event
    list is sorted by its canonical JSON representation.  Therefore two tracks
    containing the same content (same frames, events, payload values and
    timestamps) produce the same digest regardless of dict insertion order or
    the order in which the events are listed.  Timestamps are retained in the
    hash: changing an event's ``t`` changes the digest.  Duplicate events are
    retained as a multiset.

    The digest is the lowercase hex digest of the canonical UTF-8 JSON
    (``sort_keys=True``, compact separators, ``ensure_ascii=False``) that
    includes the whole track, not only the event payloads.
    """
    raise NotImplementedError("checksum() is not implemented")


def replay(track, frame_times, handler):
    """Drive ``handler`` frame by frame and return the number of events handled.

    ``frame_times`` is the frame grid in milliseconds.  Replay walks the frames
    in order and, for each frame, delivers that frame's events in ascending
    ``t`` order by calling ``handler(event)`` exactly once per event.  Every
    event must be delivered at most once within a call; multiple events in the
    same frame must not cause duplicate callbacks.

    Replay also tracks the active device: a ``device_switch`` event changes the
    active device to ``payload["device_id"]``, and every delivered event is a
    copy carrying a ``"device"`` field with the device active at that point.
    Events before the first boundary use ``track.get("device")``.

    Returns the total number of events delivered.  An empty event list returns
    ``0`` and never calls ``handler``.  Events whose ``frame`` is outside the
    supplied frame grid raise :class:`ReplayError` rather than being silently
    dropped.
    """
    raise NotImplementedError("replay() is not implemented")


def merge_tracks(tracks):
    """Merge several device tracks without losing any frame.

    Each input track has ``frame_times`` (milliseconds), ``events`` (with
    ``frame``), and optional ``device`` / ``device_priority`` fields.  Inputs
    must share the same frame-time prefix; the merged track uses the sorted
    union of all frame times, so even frames with no events are retained.

    Events from different tracks conflict when they have the same ``frame``,
    ``type`` and ``payload["key"]``.  (Events without a key conflict only when
    their complete payloads are equal.)  For a conflict, the track with the
    larger integer ``device_priority`` wins; if priorities tie, the smaller
    ``device`` string wins, then the earlier input track.  Non-conflicting
    events from every track are kept, so a lower-priority device never
    overwrites a higher-priority one.

    Returns a new track with union ``frame_times``, deterministic event order
    (frame, then ``t``), ``"truncated"`` true if any input was truncated,
    ``"device": "merged"`` and the maximum input priority.  Inputs are not
    mutated.
    """
    raise NotImplementedError("merge_tracks() is not implemented")
