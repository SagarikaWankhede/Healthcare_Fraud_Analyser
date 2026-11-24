"""Run the MediGuard web application (Flask).

    python app.py                      # http://127.0.0.1:5000
    MEDIGUARD_PORT=8000 python app.py

Artifacts must exist first: `python run_pipeline.py`.
"""
import logging
import os

from mediguard.api.app import create_app

logging.basicConfig(level=os.environ.get("MEDIGUARD_LOG_LEVEL", "INFO"),
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
app = create_app()

if __name__ == "__main__":
    app.run(host=os.environ.get("MEDIGUARD_HOST", "127.0.0.1"), port=int(os.environ.get("MEDIGUARD_PORT", "5000")),
            debug=os.environ.get("MEDIGUARD_DEBUG") == "1")
