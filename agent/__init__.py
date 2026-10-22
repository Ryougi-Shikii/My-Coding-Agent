"""LangGraph coding agent package.

Importing the package loads ``.env`` from the project root (if present) so that
every submodule can read configuration from environment variables.
"""

from pathlib import Path

from dotenv import load_dotenv

# Project root = the directory that contains app.py / cli.py / agent/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Real environment variables win over .env values (override=False).
load_dotenv(PROJECT_ROOT / ".env", override=False)
