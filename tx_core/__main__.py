"""``python -m tx_core`` → the TX CLI."""

from .cli import main

if __name__ == "__main__":
    import sys

    sys.exit(main())
