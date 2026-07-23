from __future__ import annotations


class ServiceError(Exception):
    def __init__(
        self, code: str, message: str, status_code: int = 400, details: dict | None = None
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class ModelNotFoundError(ServiceError):
    def __init__(self, model: str):
        super().__init__("model_not_found", f"model '{model}' is not registered", 404)


class InvalidInputError(ServiceError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__("invalid_input", message, 422, details)


class ServiceBusyError(ServiceError):
    def __init__(self):
        super().__init__("service_busy", "the inference worker is busy", 429)


class InferenceTimeoutError(ServiceError):
    def __init__(self, timeout_seconds: float):
        super().__init__(
            "inference_timeout",
            "inference exceeded the configured timeout",
            504,
            {"timeout_seconds": timeout_seconds},
        )
