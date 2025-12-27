"""
ServiceResponse and standard response codes used by the interactive app.

This module centralizes the response envelope and numeric/string codes so the
console I/O layer and application logic can share the same definitions without
circular imports.
"""
from dataclasses import dataclass
from typing import Optional
from enum import Enum


@dataclass
class ServiceResponse:
    """Standard response from service methods.

    code: ServiceResponse.Code enum value describing the result.
    message: optional human-readable message to show to the user.
    """

    class Code(Enum):
        """Meaningful codes for service responses.

        - OK (0): Success, no error.
        - EXIT (1): Indicates the interactive loop should exit.
        - IMAGE_LOAD_FAILED (2): Loading an image (URL/file) failed.
        - INVALID_FORMAT (3): Bad history format passed to /format.
        - CAMERA_FAILED (4): Camera capture failed.
        - NO_METRICS_SESSION (5): /metrics requested but no session available.
        - BACKEND_FAILED (6): Problems querying or switching backend.
        - UNKNOWN_COMMAND (7): Unrecognized command.
        """
        OK = 0
        EXIT = 1
        IMAGE_LOAD_FAILED = 2
        INVALID_FORMAT = 3
        CAMERA_FAILED = 4
        NO_METRICS_SESSION = 5
        BACKEND_FAILED = 6
        UNKNOWN_COMMAND = 7

    code: "ServiceResponse.Code" = Code.OK
    message: Optional[str] = None
