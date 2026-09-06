"""Version constants for TX Core (single source of truth)."""

#: Version of the TX Core engine contract (bump on breaking API changes).
__version__ = "0.1.0"

#: The engine-version stamp attached to every TX result (reproducibility).
TX_VERSION = __version__

#: Version of the Talaix Analytical Model envelope consumed by TX results.
TAM_VERSION = "1.0.0"
