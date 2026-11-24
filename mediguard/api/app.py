"""Flask application factory, error handling and request logging."""
from __future__ import annotations

import logging
import os
import time

from flask import Flask, g, jsonify, render_template, request
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException

from mediguard import __version__
from mediguard.api.db import InvestigationRepository
from mediguard.api.routes import ApiError, api
from mediguard.api.store import ArtifactStore, ArtifactsUnavailable
from mediguard.config import PATHS, Paths
from mediguard.models.inference import InputValidationError

log = logging.getLogger("mediguard.api")


def create_app(paths: Paths | None = None, store: ArtifactStore | None = None) -> Flask:
    paths = paths or PATHS
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    app = Flask(__name__, template_folder=os.path.join(root, "templates"), static_folder=os.path.join(root, "static"))
    app.config["JSON_SORT_KEYS"] = False
    app.json.sort_keys = False

    app.extensions["mediguard_store"] = store
    app.extensions["mediguard_store_error"] = None
    if store is None:
        try:
            app.extensions["mediguard_store"] = ArtifactStore.load(paths)
        except ArtifactsUnavailable as exc:
            log.error("%s", exc)
            app.extensions["mediguard_store_error"] = str(exc)
        except Exception as exc:  # corrupt artifact, version mismatch, ...
            log.exception("Failed to load artifacts")
            app.extensions["mediguard_store_error"] = f"Failed to load model artifacts: {exc}"
    app.extensions["mediguard_repo"] = InvestigationRepository(paths.db_path)
    app.register_blueprint(api)

    @app.get("/")
    def index():
        return render_template("index.html", version=__version__)

    @app.before_request
    def _start_timer():
        g.t0 = time.perf_counter()

    @app.after_request
    def _log(response):
        if request.path.startswith("/api/"):
            log.info("%s %s -> %s (%.1f ms)", request.method, request.full_path.rstrip("?"), response.status_code,
                     (time.perf_counter() - g.get("t0", time.perf_counter())) * 1000)
        return response

    @app.errorhandler(ApiError)
    def _api_error(e: ApiError):
        return jsonify({"error": e.message, "details": e.details}), e.status

    @app.errorhandler(ValidationError)
    def _validation(e: ValidationError):
        details = [{"field": ".".join(str(x) for x in err["loc"]) or "request", "message": err["msg"]}
                   for err in e.errors(include_url=False)]
        return jsonify({"error": "Invalid request", "details": details}), 422

    @app.errorhandler(InputValidationError)
    def _input_validation(e: InputValidationError):
        return jsonify({"error": "Invalid provider features", "details": e.errors}), 422

    @app.errorhandler(HTTPException)
    def _http(e: HTTPException):
        if request.path.startswith("/api/"):
            return jsonify({"error": e.description or e.name}), e.code
        return e

    @app.errorhandler(Exception)
    def _unhandled(e: Exception):
        log.exception("Unhandled error on %s", request.path)
        return jsonify({"error": "Internal server error"}), 500

    return app
