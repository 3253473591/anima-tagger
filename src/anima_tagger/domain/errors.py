class AnimaTaggerError(Exception):
    code = "ANIMA_TAGGER_ERROR"
    retryable = False


class ModelLoadError(AnimaTaggerError):
    code = "MODEL_LOAD_FAILED"


class LocalInferenceError(AnimaTaggerError):
    code = "LOCAL_INFERENCE_FAILED"


class CaptionValidationError(AnimaTaggerError):
    code = "CAPTION_VALIDATION_FAILED"

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("；".join(issues))


class CloudReviewError(AnimaTaggerError):
    code = "CLOUD_REVIEW_FAILED"

    def __init__(self, message: str, *, code: str | None = None, retryable: bool = True):
        super().__init__(message)
        self.code = code or self.code
        self.retryable = retryable


class ImageChangedError(AnimaTaggerError):
    code = "IMAGE_CHANGED"

