"""Leakage-sensitive behaviour of the feature-generating techniques."""
import numpy as np
import pandas as pd
import pytest
from mlxtend.frequent_patterns import apriori
from scipy import sparse
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split

from mediguard.features.association_rules import CodeAssociationMiner, apriori_sparse
from mediguard.features.clustering import ProviderClusterer
from mediguard.features.pipeline import FeaturePipeline
from mediguard.features.provider_features import aggregate_provider_features
from mediguard.features.regression import ReimbursementRegressor, out_of_fold_predictions
from tests.conftest import TEST_CFG


@pytest.fixture(scope="module")
def split(claims_and_labels):
    claims, labels, _ = claims_and_labels
    y = labels.set_index("Provider")["PotentialFraud"].sort_index()
    tr, te = train_test_split(y.index.to_numpy(dtype=object), test_size=0.25, stratify=y.to_numpy(), random_state=0)
    return claims, y, sorted(tr), sorted(te)


def test_apriori_sparse_matches_mlxtend():
    rng = np.random.default_rng(1)
    X = rng.random((400, 12)) < np.linspace(0.05, 0.5, 12)
    X[:, 3] |= X[:, 1] & (rng.random(400) < 0.8)
    ours = {frozenset(its): cnt for its, cnt in apriori_sparse(sparse.csc_matrix(X), 0.04, 3)}
    ref = apriori(pd.DataFrame(X, columns=range(12)), min_support=0.04, max_len=3, use_colnames=True)
    theirs = {frozenset(s): round(sup * 400) for s, sup in zip(ref["itemsets"], ref["support"])}
    assert ours == theirs


def test_rules_learned_from_training_claims_only(split):
    claims, y, tr, te = split
    tr_claims = claims[claims["Provider"].isin(tr)]
    miner = CodeAssociationMiner(min_support=0.02, min_claims=5, min_providers=3, min_lift=1.1, top_k=20)
    oof = miner.fit_transform(tr_claims, y.loc[tr])
    assert miner.train_providers_ == set(tr) and not miner.train_providers_ & set(te)
    assert miner.n_transactions_ == len(tr_claims)
    assert list(oof.index) == tr
    # rule statistics are unchanged if test providers' labels are altered (they are never seen)
    y_alt = y.copy()
    y_alt.loc[te] = 1 - y_alt.loc[te]
    miner2 = CodeAssociationMiner(min_support=0.02, min_claims=5, min_providers=3, min_lift=1.1, top_k=20)
    miner2.fit_transform(tr_claims, y_alt.loc[tr])
    pd.testing.assert_frame_equal(miner.rules_, miner2.rules_)
    assert len(miner.selected_) > 0, "planted fraud code pair should produce at least one rule"


def test_out_of_fold_rule_features_ignore_own_label(split):
    """A training provider's own label must not influence its own association feature."""
    claims, y, tr, _ = split
    tr_claims = claims[claims["Provider"].isin(tr)]
    kw = dict(min_support=0.02, min_claims=5, min_providers=3, min_lift=1.1, top_k=20, random_state=3)
    base = CodeAssociationMiner(**kw).fit_transform(tr_claims, y.loc[tr])
    target = next(p for p in tr if y[p] == 1)
    y_flip = y.loc[tr].copy()
    y_flip[target] = 0
    flipped = CodeAssociationMiner(**kw).fit_transform(tr_claims, y_flip)
    assert base.loc[target].equals(flipped.loc[target])


def test_regression_oof_is_provider_grouped(split):
    claims, _, tr, _ = split
    c = claims[claims["Provider"].isin(tr)].reset_index(drop=True)
    oof = out_of_fold_predictions(c, n_splits=3, max_iter=20)
    # Changing one provider's own targets must not change that provider's OOF predictions.
    p = tr[0]
    c2 = c.copy()
    c2.loc[c2["Provider"] == p, "InscClaimAmtReimbursed"] *= 10
    oof2 = out_of_fold_predictions(c2, n_splits=3, max_iter=20)
    mask = (c["Provider"] == p).to_numpy()
    np.testing.assert_allclose(oof[mask], oof2[mask])
    assert (oof >= 0).all()


def test_regression_unseen_categories_and_missing(split):
    claims, _, tr, te = split
    reg = ReimbursementRegressor(max_iter=20).fit(claims[claims["Provider"].isin(tr)])
    test = claims[claims["Provider"].isin(te)].copy()
    test["DiagnosisGroupCode"] = "NEVER-SEEN"
    test["DeductibleAmtPaid"] = np.nan
    pred = reg.predict(test)
    assert pred.shape == (len(test),) and np.isfinite(pred).all()


def test_dbscan_assignment_reproduces_training_labels(claims_and_labels):
    claims, _, _ = claims_and_labels
    base = aggregate_provider_features(claims)
    cl = ProviderClusterer(k_range=(2, 3), min_samples=4).fit(base)
    Z = cl._scaled(base)
    labels = DBSCAN(eps=cl.eps_, min_samples=4).fit_predict(Z)
    assigned = cl.assign(base)["dbscan_outlier"].to_numpy()
    np.testing.assert_array_equal(assigned, (labels == -1).astype(int))
    assert cl.n_clusters_ in (2, 3)


def test_pipeline_refuses_training_providers_in_transform(split):
    claims, y, tr, te = split
    fp = FeaturePipeline(TEST_CFG)
    X = fp.fit_transform(claims[claims["Provider"].isin(tr)], aggregate_provider_features(claims), y.loc[tr])
    Xt = fp.transform(claims[claims["Provider"].isin(te)], aggregate_provider_features(claims))
    assert list(X.columns) == list(Xt.columns) == fp.feature_names_
    assert "PotentialFraud" not in X.columns
    with pytest.raises(ValueError):
        fp.transform(claims[claims["Provider"].isin(tr[:3])], aggregate_provider_features(claims))
