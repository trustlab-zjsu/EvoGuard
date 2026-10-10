"""EvoGuard: oracle-free risk governance for multi-agent reasoning.

Public module boundaries:

``config``
    Typed experiment settings.
``llm``
    Model routing and request construction.
``core``
    Agent state, prompts, risk observation, and topology-strategy evolution.
``experiment``
    Episode orchestration, metrics, and output assembly.
"""

__version__ = "4.0.1"

__all__ = ["__version__"]
