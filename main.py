"""
Entry point for the Mental Health & Anxiety – DASS-21 project.

This small wrapper allows running the full pipeline with:

    python main.py

It simply delegates to `src.main.main()`, where the real logic lives.
"""

from src.main import main

if __name__ == "__main__":
    main()