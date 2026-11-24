// Run: node --test tests/js
import assert from "node:assert/strict";
import { test } from "node:test";
import { buildQuery, clamp, fmtByKind, fmtCurrency, fmtNumber, fmtPct, fmtSigned, parseRoute } from "../../static/js/format.js";

test("currency formatting", () => {
  assert.equal(fmtCurrency(67478710), "$67.48M");
  assert.equal(fmtCurrency(2_500_000_000), "$2.50B");
  assert.equal(fmtCurrency(20600), "$20.6K");
  assert.equal(fmtCurrency(1389.4), "$1,389");
  assert.equal(fmtCurrency(-130), "-$130");
  assert.equal(fmtCurrency(3212000, { compact: false }), "$3,212,000");
  assert.equal(fmtCurrency(null), "—");
  assert.equal(fmtCurrency(Number.NaN), "—");
});

test("percent, number, signed", () => {
  assert.equal(fmtPct(0.9375), "93.8%");
  assert.equal(fmtPct(0.09353, 2), "9.35%");
  assert.equal(fmtPct(undefined), "—");
  assert.equal(fmtNumber(558211), "558,211");
  assert.equal(fmtNumber(2.519, 2), "2.52");
  assert.equal(fmtSigned(0.0113), "+0.011");
  assert.equal(fmtSigned(-0.004), "-0.004");
});

test("format by feature kind", () => {
  assert.equal(fmtByKind(0.25, "ratio"), "25.0%");
  assert.equal(fmtByKind(1, "binary"), "Yes");
  assert.equal(fmtByKind(910753.76, "amount"), "$910.8K");
  assert.equal(fmtByKind(42, "count"), "42");
});

test("query string drops empty values", () => {
  assert.equal(buildQuery({ a: 1, b: "", c: null, d: undefined, e: 0, f: "x y" }), "?a=1&e=0&f=x+y");
  assert.equal(buildQuery({}), "");
});

test("hash routing", () => {
  assert.deepEqual(parseRoute("#/providers/PRV51001"), { name: "providers", param: "PRV51001" });
  assert.deepEqual(parseRoute("#/audit"), { name: "audit", param: null });
  assert.deepEqual(parseRoute(""), { name: "overview", param: null });
  assert.deepEqual(parseRoute("#/providers/score?x=1"), { name: "providers", param: "score" });
});

test("clamp", () => {
  assert.equal(clamp(500, 1, 100), 100);
  assert.equal(clamp(-3, 1, 100), 1);
});
