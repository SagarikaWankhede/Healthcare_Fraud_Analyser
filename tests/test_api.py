from dataclasses import replace

import pytest

from mediguard.api.app import create_app


@pytest.mark.parametrize("path", ["/api/health", "/api/overview", "/api/audit-simulation?capacity=5",
                                  "/api/audit-priorities?capacity=5", "/api/providers", "/api/model-performance",
                                  "/api/analysis", "/api/methodology", "/api/predict/schema", "/"])
def test_endpoints_ok(client, path):
    r = client.get(path)
    assert r.status_code == 200, r.get_data(as_text=True)[:300]


def test_overview_numbers_come_from_artifacts(client, trained):
    paths, manifest = trained
    d = client.get("/api/overview").get_json()
    assert d["model"]["version"] == manifest["model_version"]
    assert d["data"]["claims"] > 0 and 0 < d["data"]["fraud_prevalence"] < 1
    assert len(d["curves"]["k"]) == d["cohort"]["providers"]
    assert d["curves"]["exposure"][-1] == pytest.approx(1.0)


def test_simulation_recomputes_per_capacity(client):
    a = client.get("/api/audit-simulation?capacity=3").get_json()
    b = client.get("/api/audit-simulation?capacity=10").get_json()
    assert {r["providers_selected"] for r in a["strategies"]} == {3}
    assert {r["providers_selected"] for r in b["strategies"]} == {10}
    ea = next(r for r in a["strategies"] if r["strategy"] == "exposure")
    eb = next(r for r in b["strategies"] if r["strategy"] == "exposure")
    assert eb["known_fraud_reimbursement_captured"] >= ea["known_fraud_reimbursement_captured"]


@pytest.mark.parametrize("q", ["capacity=0", "capacity=abc", "capacity=100000", "capacity=5&foo=1"])
def test_simulation_validation(client, q):
    assert client.get(f"/api/audit-simulation?{q}").status_code == 422


def test_provider_list_filter_sort_paginate(client):
    full = client.get("/api/providers?per_page=100").get_json()
    n = full["total"]
    assert [p["rank"] for p in full["providers"]] == sorted(p["rank"] for p in full["providers"])
    page2 = client.get("/api/providers?per_page=4&page=2").get_json()
    assert page2["page"] == 2 and [p["rank"] for p in page2["providers"]] == [5, 6, 7, 8]
    assert page2["pages"] == -(-n // 4)
    s = client.get("/api/providers?sort=fraud_probability&order=desc&per_page=100").get_json()["providers"]
    probs = [p["fraud_probability"] for p in s]
    assert probs == sorted(probs, reverse=True)
    high = client.get("/api/providers?risk_level=high&per_page=100").get_json()["providers"]
    assert all(p["risk_level"] == "high" for p in high)
    mp = client.get("/api/providers?min_probability=0.3&per_page=100").get_json()["providers"]
    assert all(p["fraud_probability"] >= 0.3 for p in mp)
    me = client.get("/api/providers?min_exposure=1000&per_page=100").get_json()["providers"]
    assert all(p["estimated_exposure"] >= 1000 for p in me)
    cap = client.get("/api/providers?within_capacity=3").get_json()
    assert cap["total"] == 3
    one = full["providers"][0]["provider_id"]
    found = client.get(f"/api/providers?search={one[-4:]}").get_json()["providers"]
    assert any(p["provider_id"] == one for p in found)
    assert client.get("/api/providers?search=ZZZZ").get_json()["total"] == 0


@pytest.mark.parametrize("q", ["per_page=0", "per_page=1000", "sort=PotentialFraud", "order=up", "risk_level=extreme",
                               "min_probability=2", "min_probability=0.8&max_probability=0.2", "page=-1"])
def test_provider_list_validation(client, q):
    assert client.get(f"/api/providers?{q}").status_code == 422


def test_retrospective_label_hidden_by_default(client):
    rows = client.get("/api/providers").get_json()["providers"]
    assert all("retrospective_fraud_label" not in r for r in rows)
    pid = rows[0]["provider_id"]
    d = client.get(f"/api/providers/{pid}").get_json()
    assert "retrospective" not in d and "PotentialFraud" not in str(d.keys())
    retro = client.get(f"/api/providers/{pid}?evaluation_mode=retrospective").get_json()
    assert retro["retrospective"]["fraud_label"] in (0, 1)
    rows_r = client.get("/api/providers?evaluation_mode=retrospective").get_json()["providers"]
    assert all("retrospective_fraud_label" in r for r in rows_r)


def test_provider_profile_and_unknown_ids(client):
    pid = client.get("/api/providers").get_json()["providers"][0]["provider_id"]
    d = client.get(f"/api/providers/{pid}").get_json()
    for key in ("scores", "recommendation", "billing", "trend", "claim_mix", "anomaly", "explanation", "investigation"):
        assert key in d
    assert d["scores"]["rank_exposure"] == 1
    assert sum(t["claims"] for t in d["trend"]) == d["billing"]["total_claims"]
    assert client.get("/api/providers/PRV_DOES_NOT_EXIST").status_code == 404
    assert client.get("/api/providers/PRV_DOES_NOT_EXIST/status").status_code == 404


def test_status_persistence_and_history(client):
    pid = client.get("/api/providers").get_json()["providers"][1]["provider_id"]
    assert client.get(f"/api/providers/{pid}/status").get_json()["status"] == "not_reviewed"
    r = client.patch(f"/api/providers/{pid}/status", json={"status": "under_review", "note": "opened"})
    assert r.status_code == 200
    client.patch(f"/api/providers/{pid}/status", json={"status": "escalated"})
    rec = client.get(f"/api/providers/{pid}/status").get_json()
    assert rec["status"] == "escalated" and [h["status"] for h in rec["history"]] == ["escalated", "under_review"]
    assert client.get("/api/providers?status=escalated").get_json()["total"] == 1
    assert client.patch(f"/api/providers/{pid}/status", json={"status": "fraud"}).status_code == 422
    assert client.patch(f"/api/providers/{pid}/status", data="not json", content_type="application/json").status_code == 400
    assert client.patch(f"/api/providers/{pid}/status", json={"status": "closed", "extra": 1}).status_code == 422


def test_predict_endpoint(client):
    schema = client.get("/api/predict/schema").get_json()
    for e in schema["examples"]:
        assert isinstance(client.post("/api/predict", json={"features": e["features"]}).get_json()["contributions"], list)
    ex = schema["examples"][1]  # median example: not saturated, so substitutions move the estimate
    r = client.post("/api/predict", json={"features": ex["features"]})
    assert r.status_code == 200
    out = r.get_json()
    assert 0 <= out["fraud_probability"] <= 1
    assert out["audit_priority"] in ("P1", "P2", "P3", "Routine")
    assert out["contributions"] and out["model_version"] == schema["model_version"]
    assert out["estimated_exposure"] == pytest.approx(out["fraud_probability"] * ex["features"]["total_reimbursed"])


@pytest.mark.parametrize("body,status", [
    ({"features": {"total_claims": 5, "total_reimbursed": 100, "unique_beneficiaries": 9}}, 422),
    ({"features": {"total_claims": "x", "total_reimbursed": 100, "unique_beneficiaries": 2}}, 422),
    ({"features": {}}, 422),
    ({"wrong": {}}, 422),
    ([1, 2], 400),
])
def test_predict_validation(client, body, status):
    r = client.post("/api/predict", json=body)
    assert r.status_code == status
    assert "error" in r.get_json()


def test_missing_artifacts_return_503(tmp_path, trained):
    paths = replace(trained[0], artifact_dir=tmp_path / "empty", db_path=tmp_path / "x.db")
    app = create_app(paths)
    c = app.test_client()
    h = c.get("/api/health")
    assert h.status_code == 503 and h.get_json()["model_loaded"] is False
    r = c.get("/api/overview")
    assert r.status_code == 503 and "run_pipeline" in r.get_json()["error"]
    assert c.post("/api/predict", json={"features": {"total_claims": 1}}).status_code == 503
    assert c.get("/").status_code == 200  # the shell still loads and shows the error state


def test_unknown_api_route_is_json_404(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.get_json()["error"]
