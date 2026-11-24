import logging
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mediguard.config import TRAINING, Paths  # noqa: E402
from tests.synthetic import write_synthetic_raw  # noqa: E402

TEST_CFG = replace(
    TRAINING, cv_folds=3, inner_folds=3, regression_max_iter=40, cluster_k_range=(2, 3, 4),
    dbscan_min_samples=4, apriori_min_support=0.02, rule_min_claims=5, rule_min_providers=3,
    rule_min_lift=1.1, rule_top_k=20, random_baseline_simulations=200,
)


@pytest.fixture(scope="session")
def synthetic_paths(tmp_path_factory) -> Paths:
    root = tmp_path_factory.mktemp("mediguard")
    paths = Paths(raw_dir=root / "raw", processed_dir=root / "processed", artifact_dir=root / "artifacts",
                  db_path=root / "investigations.db")
    write_synthetic_raw(paths.raw_dir)
    return paths


@pytest.fixture(scope="session")
def trained(synthetic_paths):
    """Run the real training pipeline once on synthetic data (quick grids)."""
    from mediguard.training.train import run
    logging.getLogger("mediguard").setLevel(logging.WARNING)
    manifest = run(synthetic_paths, TEST_CFG, quick=True)
    return synthetic_paths, manifest


@pytest.fixture(scope="session")
def store(trained):
    from mediguard.api.store import ArtifactStore
    return ArtifactStore.load(trained[0])


@pytest.fixture()
def client(trained, store, tmp_path):
    from mediguard.api.app import create_app
    paths = replace(trained[0], db_path=tmp_path / "inv.db")
    app = create_app(paths, store=store)
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture(scope="session")
def claims_and_labels(synthetic_paths):
    from mediguard.data.ingest import load_raw_tables
    from mediguard.data.preprocess import build_claims
    return build_claims(load_raw_tables(synthetic_paths.raw_dir))
