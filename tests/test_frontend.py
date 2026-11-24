"""Browser tests (Playwright) against a live server on synthetic artifacts, plus the JS unit tests.

Uses an installed Microsoft Edge or Google Chrome; set MEDIGUARD_BROWSER_CHANNEL to choose
(default: msedge, then chrome, then Playwright's bundled chromium). Skipped if none is available.
"""
import os
import shutil
import socket
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_js_unit_tests():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    r = subprocess.run([node, "--test", str(ROOT / "tests" / "js" / "format.test.mjs")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_js_modules_parse():
    """Every ES module must parse as a module (a syntax error blanks the whole app)."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    for f in (ROOT / "static" / "js").rglob("*.js"):
        r = subprocess.run([node, "--input-type=module", "--check"], input=f.read_text(encoding="utf-8"),
                           capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 0, f"{f}: {r.stderr}"


playwright = pytest.importorskip("playwright.sync_api")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(trained, store, tmp_path_factory):
    from werkzeug.serving import make_server
    from mediguard.api.app import create_app
    paths = replace(trained[0], db_path=tmp_path_factory.mktemp("e2e") / "inv.db")
    app = create_app(paths, store=store)
    port = _free_port()
    srv = make_server("127.0.0.1", port, app, threaded=True)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


@pytest.fixture(scope="module")
def browser():
    pw = playwright.sync_playwright().start()
    b = None
    for channel in [os.environ.get("MEDIGUARD_BROWSER_CHANNEL"), "msedge", "chrome", None]:
        try:
            b = pw.chromium.launch(channel=channel) if channel else pw.chromium.launch()
            break
        except Exception:
            continue
    if b is None:
        pw.stop()
        pytest.skip("no Chromium-based browser available")
    yield b
    b.close()
    pw.stop()


@pytest.fixture()
def page(browser):
    ctx = browser.new_context(viewport={"width": 1366, "height": 900})
    pg = ctx.new_page()
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("console", lambda m: pg.errors.append(m.text) if m.type == "error" else None)
    yield pg
    ctx.close()


def test_navigation_and_no_console_errors(page, server):
    page.goto(f"{server}/#/overview")
    page.wait_for_selector("[data-kpi=claims] .value")
    for nav, heading in [("audit", "Audit Priorities"), ("providers", "Provider Analysis"),
                         ("performance", "Model Performance"), ("methodology", "Methodology"), ("overview", "Overview")]:
        page.click(f"[data-nav={nav}]")
        page.wait_for_selector(f"h1:has-text('{heading}')")
        assert page.get_attribute(f"[data-nav={nav}]", "aria-current") == "page"
        page.wait_for_load_state("networkidle")
    assert not page.errors, page.errors


def test_capacity_slider_and_presets_update_comparison(page, server):
    page.goto(f"{server}/#/audit")
    page.wait_for_selector("[data-testid=queue-total]")
    page.click("[data-capacity='10']")
    page.wait_for_function("document.querySelector('[data-testid=capacity-value]').textContent === '10'")
    page.wait_for_function("document.querySelector('[data-testid=queue-total]').textContent.startsWith('10 ')")
    slider = page.locator("[data-testid=capacity-slider]")
    slider.fill("5")
    page.wait_for_function("document.querySelector('[data-testid=capacity-value]').textContent === '5'")
    page.wait_for_function("document.querySelector('[data-testid=queue-total]').textContent.startsWith('5 ')")
    cells = page.locator("section:has([data-testid=compare-title]) tbody tr").first.locator("td").all_inner_texts()
    assert cells[1].strip() != ""


def test_queue_filtering_and_pagination(page, server):
    page.goto(f"{server}/#/audit")
    page.wait_for_selector("[data-testid=queue-total]")
    page.uncheck("[data-testid=only-capacity]")
    page.wait_for_load_state("networkidle")
    page.select_option("[data-testid=per-page]", "10")
    page.wait_for_function("document.querySelector('[data-testid=page-indicator]').textContent.startsWith('Page 1 of')")
    first_page = page.locator("[data-testid=queue] tbody tr td:nth-child(2)").all_inner_texts()
    page.click("[data-testid=next-page]")
    page.wait_for_function("document.querySelector('[data-testid=page-indicator]').textContent.startsWith('Page 2')")
    assert page.locator("[data-testid=queue] tbody tr td:nth-child(2)").all_inner_texts() != first_page
    target = first_page[0].strip()
    page.fill("[data-testid=search]", target)
    page.wait_for_function("document.querySelector('[data-testid=queue-total]').textContent.startsWith('1 ')")
    page.fill("[data-testid=search]", "NO-SUCH-PROVIDER")
    page.wait_for_selector("text=No records match the current filters.")
    page.fill("[data-testid=search]", "")
    page.select_option("[data-filter=risk_level]", "high")
    page.wait_for_load_state("networkidle")
    risks = page.locator("[data-testid=queue] tbody tr td:nth-child(9)").all_inner_texts()
    assert all(r.strip() == "High" for r in risks if r.strip())


def test_provider_profile_status_and_retrospective_mode(page, server):
    page.goto(f"{server}/#/audit")
    page.wait_for_selector("[data-testid=queue] tbody tr.clickable")
    page.locator("[data-testid=queue] tbody tr.clickable").first.click()
    page.wait_for_selector("[data-testid=profile-id]")
    assert page.locator("[data-testid=retro-panel]").count() == 0
    page.select_option("[data-testid=status-select]", "escalated")
    page.click("[data-testid=status-save]")
    page.wait_for_selector("[data-testid=status-msg]:has-text('Saved: Escalated')")
    page.check("[data-testid=retro-toggle]")
    page.wait_for_selector("[data-testid=retro-panel]")
    page.reload()
    page.wait_for_selector("[data-testid=status-select]")
    assert page.input_value("[data-testid=status-select]") == "escalated"


def test_prediction_form(page, server):
    page.goto(f"{server}/#/providers/score")
    page.wait_for_selector("[data-example]")
    page.click("[data-testid=predict-submit]")  # empty form -> client-side required errors
    page.wait_for_selector("[data-testid=predict-error]")
    page.locator("[data-example]").nth(1).click()
    page.click("[data-testid=predict-submit]")
    page.wait_for_selector("[data-testid=predict-result]")
    prob = page.inner_text("[data-testid=predicted-probability]")
    assert prob.endswith("%") and 0 <= float(prob[:-1]) <= 100
    # server-side validation: beneficiaries cannot exceed claims
    page.fill("[data-feature=total_claims]", "3")
    page.fill("[data-feature=unique_beneficiaries]", "50")
    page.click("[data-testid=predict-submit]")
    page.wait_for_selector("[data-field=unique_beneficiaries].invalid")
    page.click("[data-testid=reset]")
    assert page.input_value("[data-feature=total_claims]") == ""


def test_loading_and_api_error_states(page, server):
    page.route("**/api/overview", lambda route: route.fulfill(status=503, content_type="application/json",
                                                              body='{"error": "Model artifacts missing"}'))
    page.goto(f"{server}/#/overview")
    page.wait_for_selector(".error-box:has-text('Model artifacts missing')")
    page.unroute("**/api/overview")

    def slow(route):
        import time
        time.sleep(0.8)
        route.continue_()
    page.route("**/api/methodology", slow)
    page.goto(f"{server}/#/methodology")
    page.wait_for_selector(".loading")
    page.wait_for_selector("h2:has-text('Pipeline')")


def test_mobile_layout_has_no_horizontal_scroll(browser, server):
    ctx = browser.new_context(viewport={"width": 390, "height": 844})
    pg = ctx.new_page()
    for route in ["overview", "audit", "providers/score", "methodology"]:
        pg.goto(f"{server}/#/{route}")
        pg.wait_for_load_state("networkidle")
        pg.wait_for_timeout(200)
        assert pg.evaluate("document.documentElement.scrollWidth") <= 392, route
    ctx.close()
