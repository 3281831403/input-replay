"""Deterministic replay primitives for recorded input events.

All timestamps are integer milliseconds.  Frames are described by a strictly
increasing list of frame start times, so a recording may use a variable frame
rate.  Events have the shape::

    {"t": int, "type": str, "payload": dict}

Every public entry point is deterministic: no randomness, no wall-clock time,
and no dependence on dict insertion order.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left, bisect_right

MAX_EVENTS = 10_000
DEVICE_SWITCH = "device_switch"

_JSON_KWARGS = {
    "sort_keys": True,
    "separators": (",", ":"),
    "ensure_ascii": False,
}


class ReplayError(Exception):
    """Raised when a recording does not satisfy the replay contract."""


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_frame_times(frame_times, *, require_any=True):
    """Validate and return a copied list of strictly increasing frame times."""
    if not isinstance(frame_times, (list, tuple)):
        raise ReplayError("frame_times must be a list of integers")
    frames = list(frame_times)
    if require_any and not frames:
        raise ReplayError("frame_times must contain at least one frame")
    previous = None
    for frame_time in frames:
        if not _is_int(frame_time):
            raise ReplayError("frame times must be integers")
        if previous is not None and frame_time <= previous:
            raise ReplayError("frame_times must be strictly increasing")
        previous = frame_time
    return frames


def _canonical(value):
    """Canonical JSON used for stable comparison and ordering."""
    return json.dumps(value, **_JSON_KWARGS)


def _copy_event(event, *, copy_payload=True):
    copied = dict(event)
    if copy_payload and isinstance(copied.get("payload"), dict):
        copied["payload"] = dict(copied["payload"])
    return copied


def _event_frame_bounds(event, frame_count):
    frame = event.get("frame")
    if not _is_int(frame):
        raise ReplayError("event is missing an integer 'frame'")
    if frame < 0 or frame >= frame_count:
        raise ReplayError(f"event frame {frame} is outside the frame grid")
    return frame


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
    frames = _validate_frame_times(frame_times)
    if not isinstance(events, (list, tuple)):
        raise ReplayError("events must be a list of event mappings")

    last_frame = len(frames) - 1
    archived = []
    truncated = False
    previous_t = None

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ReplayError("each event must be a mapping")
        t = event.get("t")
        if not _is_int(t):
            raise ReplayError("event 't' must be an integer")
        event_type = event.get("type")
        if not isinstance(event_type, str):
            raise ReplayError("event 'type' must be a string")
        if not isinstance(event.get("payload"), dict):
            raise ReplayError("event 'payload' must be a mapping")
        if previous_t is not None and t < previous_t:
            raise ReplayError("event timestamps must not go backwards")
        previous_t = t

        if index >= MAX_EVENTS:
            truncated = True
            continue

        frame = bisect_right(frames, t) - 1
        if frame < 0:
            frame = 0
        elif frame > last_frame:
            frame = last_frame

        copied = _copy_event(event)
        copied["frame"] = frame
        archived.append(copied)

    return {
        "frame_times": frames,
        "events": archived,
        "truncated": truncated,
    }


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
    if not _is_number(tolerance) or not math.isfinite(tolerance):
        raise ReplayError("tolerance must be a finite number")
    if tolerance < 0:
        raise ReplayError("tolerance must be non-negative")
    frames = _validate_frame_times(frame_times, require_any=False)
    if not isinstance(events, (list, tuple)):
        raise ReplayError("events must be a list of event mappings")
    if not frames and events:
        raise ReplayError("frame_times must contain at least one frame")

    aligned = []
    for event in events:
        if not isinstance(event, dict):
            raise ReplayError("each event must be a mapping")
        t = event.get("t")
        if not _is_int(t):
            raise ReplayError("event 't' must be an integer")

        position = bisect_left(frames, t)
        if position <= 0:
            frame = 0
        elif position >= len(frames):
            frame = len(frames) - 1
        else:
            before = frames[position - 1]
            after = frames[position]
            # A tie between neighbours goes to the later frame.
            frame = position if (after - t) <= (t - before) else position - 1

        delta_ms = t - frames[frame]
        if abs(delta_ms) > tolerance:
            continue

        copied = _copy_event(event)
        copied["frame"] = frame
        copied["delta_ms"] = delta_ms
        aligned.append(copied)

    return aligned


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
    if not isinstance(events, (list, tuple)):
        raise ReplayError("events must be a list of event mappings")

    group_keys = []
    last_index = {}

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ReplayError("each event must be a mapping")
        frame = event.get("frame")
        if not _is_int(frame):
            raise ReplayError("event is missing an integer 'frame'")
        event_type = event.get("type")
        if not isinstance(event_type, str):
            raise ReplayError("event is missing a string 'type'")

        payload = event.get("payload")
        if isinstance(payload, dict) and "key" in payload:
            key = (frame, event_type, _canonical(payload["key"]))
            last_index[key] = index
        else:
            key = None
        group_keys.append(key)

    kept = []
    for index, event in enumerate(events):
        key = group_keys[index]
        if key is None or last_index[key] == index:
            kept.append(event)
    return kept


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
    if not _is_number(axis) or not math.isfinite(axis):
        raise ReplayError("axis must be a finite number")
    if not _is_number(deadzone) or not math.isfinite(deadzone):
        raise ReplayError("deadzone must be a finite number")
    if not _is_number(sensitivity) or not math.isfinite(sensitivity):
        raise ReplayError("sensitivity must be a finite number")
    if deadzone < 0.0 or deadzone >= 1.0:
        raise ReplayError("deadzone must be in [0.0, 1.0)")
    if sensitivity < 0.0:
        raise ReplayError("sensitivity must be non-negative")
    if axis < -1.0 or axis > 1.0:
        raise ReplayError("axis must be in [-1.0, 1.0]")

    magnitude = abs(axis)
    if magnitude <= deadzone:
        return 0.0

    normalized = (magnitude - deadzone) / (1.0 - deadzone) * sensitivity
    if axis < 0.0:
        return -normalized
    return normalized


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
    if not isinstance(track, dict):
        raise ReplayError("track must be a mapping")

    switched = dict(track)
    events = list(track.get("events") or [])
    copied_events = [_copy_event(event) for event in events]

    if events:
        last = events[-1]
        if not _is_int(last.get("t")):
            raise ReplayError("last event is missing an integer 't'")
        boundary_t = last["t"] + 1
        boundary_frame = last.get("frame", 0)
        if not _is_int(boundary_frame):
            raise ReplayError("last event is missing an integer 'frame'")
    else:
        frames = list(track.get("frame_times") or [])
        if frames:
            boundary_t = frames[-1]
            boundary_frame = len(frames) - 1
        else:
            boundary_t = 0
            boundary_frame = 0

    boundary = {
        "t": boundary_t,
        "type": DEVICE_SWITCH,
        "payload": {
            "device_id": device_id,
            "from": track.get("device"),
        },
        "frame": boundary_frame,
    }
    copied_events.append(boundary)

    switched["events"] = copied_events
    switched["device"] = device_id
    return switched


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
    if not isinstance(track, dict):
        raise ReplayError("track must be a mapping")

    canonical_track = dict(track)
    if "events" in canonical_track:
        events = canonical_track["events"]
        if not isinstance(events, (list, tuple)):
            raise ReplayError("track events must be a list")
        canonical_track["events"] = sorted(events, key=_canonical)

    blob = _canonical(canonical_track).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


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
    if not isinstance(track, dict):
        raise ReplayError("track must be a mapping")
    if not callable(handler):
        raise ReplayError("handler must be callable")

    events = track.get("events") or []
    if not isinstance(events, (list, tuple)):
        raise ReplayError("track events must be a list")
    if not events:
        return 0

    frames = _validate_frame_times(frame_times)

    by_frame = {}
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ReplayError("each event must be a mapping")
        frame = _event_frame_bounds(event, len(frames))
        if not _is_int(event.get("t")):
            raise ReplayError("event 't' must be an integer")
        by_frame.setdefault(frame, []).append((index, event))

    active_device = track.get("device")
    delivered = 0

    for frame in range(len(frames)):
        # Stable sort by time; ties keep archive order.
        batch = sorted(
            by_frame.get(frame, ()),
            key=lambda item: (item[1]["t"], item[0]),
        )
        for _, event in batch:
            if event.get("type") == DEVICE_SWITCH:
                payload = event.get("payload")
                if isinstance(payload, dict) and "device_id" in payload:
                    active_device = payload["device_id"]

            copied = _copy_event(event)
            copied["device"] = active_device
            handler(copied)
            delivered += 1

    return delivered


def _merge_conflict_key(event):
    """Identity used to decide whether two events from different tracks clash."""
    payload = event.get("payload")
    if isinstance(payload, dict) and "key" in payload:
        return (event["frame"], event["type"], "key", _canonical(payload["key"]))
    return (event["frame"], event["type"], "payload", _canonical(payload))


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
    if not isinstance(tracks, (list, tuple)):
        raise ReplayError("tracks must be a list of track mappings")

    frames_per_track = []
    for track in tracks:
        if not isinstance(track, dict):
            raise ReplayError("each track must be a mapping")
        frames_per_track.append(
            _validate_frame_times(track.get("frame_times") or [], require_any=False)
        )

    # Every input must agree on the common frame-time prefix, otherwise the
    # same frame index would mean different times in different tracks.
    for i in range(len(frames_per_track)):
        for j in range(i + 1, len(frames_per_track)):
            left = frames_per_track[i]
            right = frames_per_track[j]
            common = min(len(left), len(right))
            if left[:common] != right[:common]:
                raise ReplayError("tracks do not share a frame-time prefix")

    union_times = sorted(
        {frame_time for frames in frames_per_track for frame_time in frames}
    )
    union_index = {frame_time: index for index, frame_time in enumerate(union_times)}

    priorities = []
    for track in tracks:
        priority = track.get("device_priority", 0)
        if not _is_int(priority):
            raise ReplayError("device_priority must be an integer")
        priorities.append(priority)

    entries = []
    for track_index, (track, frames) in enumerate(zip(tracks, frames_per_track)):
        events = track.get("events") or []
        if not isinstance(events, (list, tuple)):
            raise ReplayError("track events must be a list")
        for event in events:
            if not isinstance(event, dict):
                raise ReplayError("each event must be a mapping")
            frame = _event_frame_bounds(event, len(frames))
            if not _is_int(event.get("t")):
                raise ReplayError("event 't' must be an integer")
            copied = _copy_event(event)
            copied["frame"] = union_index[frames[frame]]
            entries.append(
                {
                    "track": track_index,
                    "order": len(entries),
                    "event": copied,
                }
            )

    # Resolve cross-track conflicts.  Events from the same track never
    # conflict with one another; only the winning track's events survive.
    groups = {}
    for entry in entries:
        key = _merge_conflict_key(entry["event"])
        groups.setdefault(key, []).append(entry)

    keep = set()
    for group in groups.values():
        track_numbers = {entry["track"] for entry in group}
        if len(track_numbers) < 2:
            keep.update(entry["order"] for entry in group)
            continue

        winner = min(
            group,
            key=lambda entry: (
                -priorities[entry["track"]],
                str(tracks[entry["track"]].get("device", "")),
                entry["track"],
            ),
        )["track"]
        keep.update(entry["order"] for entry in group if entry["track"] == winner)

    merged_events = [
        entry["event"] for entry in entries if entry["order"] in keep
    ]
    merged_events.sort(key=lambda event: (event["frame"], event["t"]))

    return {
        "frame_times": union_times,
        "events": merged_events,
        "truncated": any(bool(track.get("truncated", False)) for track in tracks),
        "device": "merged",
        "device_priority": max(priorities, default=0),
    }
