"""Outbound instructor hook. Nothing is published in this version.

A later publisher can subclass this and register it on the agent. Callbacks must
not move the arm or open a second camera client.
"""


class InstructorHook:
    enabled = False

    def on_status(self, status: dict) -> None:
        return None

    def on_frame(self, jpeg: bytes) -> None:
        return None

    def on_detections(self, boxes: list) -> None:
        return None


class NullHook(InstructorHook):
    """The installed hook. It discards every event."""
