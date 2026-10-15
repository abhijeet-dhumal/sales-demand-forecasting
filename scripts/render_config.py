#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import render_project


if __name__ == "__main__":
    output = render_project()
    print(f"Rendered project configuration to {output}")
