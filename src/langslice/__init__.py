"""LangSlice: VLM-based brain slice registration."""
__version__ = "0.1.0"

# Tool-calling Gemini models (especially Flash) routinely return responses
# whose only part is a `function_call` — when google-genai's response.text
# property is accessed under those conditions, it emits a noisy WARNING via
# the `google_genai.types` logger. Our agent loop reads function calls via
# `event.get_function_calls()` directly so the warning is pure noise, but
# it leaks all the way out to the Tauri agent panel and looks like an
# error. Silence just that specific message; other google_genai warnings
# stay visible.
import logging as _logging


class _GeminiNonTextPartsFilter(_logging.Filter):
    def filter(self, record: _logging.LogRecord) -> bool:
        return "non-text parts in the response" not in record.getMessage()


_logging.getLogger("google_genai.types").addFilter(_GeminiNonTextPartsFilter())
