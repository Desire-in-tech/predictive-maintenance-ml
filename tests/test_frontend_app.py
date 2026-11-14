from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_runtime_errors():
    app = AppTest.from_file("frontend/app.py").run()
    assert len(app.exception) == 0
    assert any("Industrial Predictive Maintenance" in item.value for item in app.title)
