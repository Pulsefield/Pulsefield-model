import pytest

from pulsefield_model.inference.service_models import (
    AudioCommand,
    EndOfStreamEvent,
    ErrorEvent,
    HitObjectTokenEvent,
    MapperStreamBeginEvent,
    ReferenceTimeCommand,
    StatusEvent,
    StopCommand,
    command_to_endpoint_payload,
    event_to_endpoint_payload,
    service_command_from_endpoint_payload,
    service_event_from_endpoint_payload,
)


def test_audio_command_converts_from_and_to_endpoint_payload() -> None:
    payload = {
        "type": "audio",
        "session_id": "s1",
        "audio_path": "/tmp/song.wav",
        "audio_length_ms": 2_000,
        "difficulty": 5,
        "route": "INFERENCE_ROUTE_TIMING_MOCK",
    }

    command = service_command_from_endpoint_payload(payload)

    assert command == AudioCommand(
        session_id="s1",
        audio_path="/tmp/song.wav",
        audio_length_ms=2_000,
        difficulty=5.0,
        route="timing_mock",
    )
    assert command_to_endpoint_payload(command) == {
        "type": "audio",
        "session_id": "s1",
        "audio_path": "/tmp/song.wav",
        "audio_length_ms": 2_000,
        "difficulty": 5.0,
        "route": "timing_mock",
    }


def test_command_payload_serialization_omits_absent_optionals() -> None:
    assert command_to_endpoint_payload(AudioCommand(session_id="s1", audio_path="/tmp/song.wav")) == {
        "type": "audio",
        "session_id": "s1",
        "audio_path": "/tmp/song.wav",
        "route": "mapper",
    }
    assert command_to_endpoint_payload(
        ReferenceTimeCommand(session_id="s1", ref_time_ms=1_234, local_host_time_send_ms=50_000.25)
    ) == {
        "type": "reference_time",
        "session_id": "s1",
        "ref_time_ms": 1_234,
        "local_host_time_send_ms": 50_000.25,
    }
    assert service_command_from_endpoint_payload({"type": "stop", "session_id": "s1"}) == StopCommand(
        session_id="s1"
    )


def test_stream_events_convert_from_and_to_endpoint_payloads() -> None:
    cases = [
        (
            {
                "type": "mapper_stream_begin",
                "session_id": "s1",
                "token_contract_version": 2,
                "audio_length_ms": 2_000,
            },
            MapperStreamBeginEvent(session_id="s1", token_contract_version=2, audio_length_ms=2_000),
        ),
        (
            {
                "type": "hit_object_token",
                "session_id": "s1",
                "token_id": 10,
                "ms_in_ref_audio": 1_240,
                "token_index": 3,
            },
            HitObjectTokenEvent(session_id="s1", token_id=10, ms_in_ref_audio=1_240, token_index=3),
        ),
        (
            {
                "type": "end_of_stream",
                "session_id": "s1",
                "complete_through_ms": 2_000,
                "audio_length_ms": 2_000,
            },
            EndOfStreamEvent(session_id="s1", complete_through_ms=2_000, audio_length_ms=2_000),
        ),
    ]

    for payload, event in cases:
        assert service_event_from_endpoint_payload(payload) == event
        assert event_to_endpoint_payload(event) == payload


def test_error_and_status_events_preserve_endpoint_compatible_fields() -> None:
    error_payload = {
        "type": "error",
        "code": "audio_not_found",
        "message": "missing file",
        "error": "missing file",
        "session_id": "s1",
        "phase": "prepare",
        "route": "INFERENCE_ROUTE_MAPPER",
    }
    status_payload = {
        "type": "status",
        "to": "audio_ready",
        "from": "audio_preparing",
        "message": "prepared",
        "reason": "audio_prepared",
        "session_id": "s1",
        "route": "timing-mock",
        "ref_time_ms": 1_234,
        "difficulty": 4,
    }

    assert service_event_from_endpoint_payload(error_payload) == ErrorEvent(
        code="audio_not_found",
        message="missing file",
        error="missing file",
        session_id="s1",
        phase="prepare",
        route="mapper",
    )
    status_event = service_event_from_endpoint_payload(status_payload)
    assert status_event == StatusEvent(
        status="audio_ready",
        from_status="audio_preparing",
        message="prepared",
        reason="audio_prepared",
        session_id="s1",
        route="timing_mock",
        ref_time_ms=1_234,
        difficulty=4.0,
    )
    assert event_to_endpoint_payload(status_event) == {
        "type": "status",
        "message": "prepared",
        "reason": "audio_prepared",
        "session_id": "s1",
        "status": "audio_ready",
        "from_status": "audio_preparing",
        "route": "timing_mock",
        "ref_time_ms": 1_234,
        "difficulty": 4.0,
    }


def test_payload_validation_rejects_unsupported_shapes() -> None:
    with pytest.raises(ValueError, match="unsupported service command type"):
        service_command_from_endpoint_payload({"type": "unknown"})
    with pytest.raises(ValueError, match="audio_length_ms must be an integer"):
        service_command_from_endpoint_payload({
            "type": "audio",
            "session_id": "s1",
            "audio_path": "/tmp/song.wav",
            "audio_length_ms": 1.5,
        })
    with pytest.raises(ValueError, match="audio_length_ms must be positive"):
        service_command_from_endpoint_payload({
            "type": "audio",
            "session_id": "s1",
            "audio_path": "/tmp/song.wav",
            "audio_length_ms": 0,
        })
    with pytest.raises(ValueError, match="audio_length_ms must be positive"):
        service_command_from_endpoint_payload({
            "type": "audio",
            "session_id": "s1",
            "audio_path": "/tmp/song.wav",
            "audio_length_ms": -1,
        })
    with pytest.raises(ValueError, match="session_id must be a string"):
        service_command_from_endpoint_payload({"type": "audio", "session_id": 123})
    with pytest.raises(ValueError, match="session_id must be a non-empty string"):
        service_command_from_endpoint_payload({"type": "audio", "session_id": "", "audio_path": "/tmp/song.wav"})
    with pytest.raises(ValueError, match="audio_path must be a string"):
        service_command_from_endpoint_payload({"type": "audio", "session_id": "s1", "audio_path": 123})
    with pytest.raises(ValueError, match="audio_path must be a string"):
        service_command_from_endpoint_payload({"type": "audio", "session_id": "s1"})
    with pytest.raises(ValueError, match="audio_path must be a non-empty string"):
        service_command_from_endpoint_payload({"type": "audio", "session_id": "s1", "audio_path": " "})
    with pytest.raises(ValueError, match="ref_time_ms must be non-negative"):
        service_command_from_endpoint_payload({
            "type": "reference_time",
            "session_id": "s1",
            "ref_time_ms": -1,
            "local_host_time_send_ms": 50_000,
        })
    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        service_command_from_endpoint_payload({"type": "stop", "session_id": "s1", "reason": ""})
    with pytest.raises(ValueError, match="route must be a string"):
        service_command_from_endpoint_payload({
            "type": "audio",
            "session_id": "s1",
            "audio_path": "/tmp/song.wav",
            "route": 123,
        })
    with pytest.raises(ValueError, match="unsupported service event type"):
        service_event_from_endpoint_payload({"type": "unknown"})
    with pytest.raises(ValueError, match="session_id must be a string"):
        service_event_from_endpoint_payload({"type": "hit_object_token", "token_id": 10, "ms_in_ref_audio": 1_240})
    with pytest.raises(ValueError, match="token_id must be non-negative"):
        service_event_from_endpoint_payload({
            "type": "hit_object_token",
            "session_id": "s1",
            "token_id": -1,
            "ms_in_ref_audio": 1_240,
        })
    with pytest.raises(ValueError, match="ms_in_ref_audio must be non-negative"):
        service_event_from_endpoint_payload({
            "type": "hit_object_token",
            "session_id": "s1",
            "token_id": 10,
            "ms_in_ref_audio": -1,
        })
