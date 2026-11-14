from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_runtime_errors():
    app_path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    app = AppTest.from_file(str(app_path)).run()
    assert len(app.exception) == 0
    assert any("Industrial Predictive Maintenance" in item.value for item in app.title)
