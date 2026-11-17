"""让 tests/ 能 import 根目录的 llm.py、config.py。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
