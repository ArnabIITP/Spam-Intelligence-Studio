import json
from pathlib import Path


def test_report_notebook_uses_package_imports():
    notebook_path = Path("notebooks") / "spam_intelligence_report.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "spam_intelligence" in sources
