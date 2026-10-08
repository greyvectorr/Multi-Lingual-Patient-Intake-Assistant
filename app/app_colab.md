# `app_colab.py` — Colab launcher

```python
import app
app.demo.launch(share=True)
```

Imports `app` (which builds the Gradio `demo` object) and launches it with a public share link, which is what Colab needs to expose the UI.

```python
!python app_colab.py
```

Notes:
- The share link is temporary (Gradio says up to ~1 week) and **publicly reachable** — don't use real patient data on it.
- Colab runtimes are ephemeral: models re-download and the SQLite database is lost when the runtime resets (see the database-location note in [clinical_note.md](clinical_note.md)).
- It does nothing else. In particular it does not pre-load models, so the first intake after launch is slow.
