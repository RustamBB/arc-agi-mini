"""arcgen: synthetic ARC-AGI-like task generator built from a bank of real grid operations."""
from .ops import OPS  # noqa: F401
from .program import run, to_text  # noqa: F401
from .dataset import generate, make_task  # noqa: F401
