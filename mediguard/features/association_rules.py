"""Technique 3 - Association rule mining on diagnosis/procedure code combinations.

Each claim is a transaction whose items are its diagnosis codes (``D_<icd9>``) and procedure
codes (``P_<icd9-proc>``).

1. **Apriori** (:func:`apriori_sparse`) finds frequent itemsets on *training claims only*. This step is
   unsupervised. Code-to-code rules (e.g. ``{D_4019} -> {D_25000}``) are reported with
   support/confidence/lift to describe normal coding patterns.
2. **Class association rules** ``{itemset} -> {claim billed by a fraud-labelled provider}`` are
   scored on training claims:
     * support    = P(itemset AND fraud-labelled provider)
     * confidence = P(fraud-labelled provider | itemset)
     * lift       = confidence / P(fraud-labelled provider)   (>1 = over-represented)
   Rules must pass minimum claim, provider and fraud-provider counts so that one large provider
   cannot create a "pattern" on its own. A high lift is a statistical association with
   historically labelled providers, not evidence that a claim is fraudulent; a frequent
   itemset on its own says nothing about fraud.
3. Provider features: number/share of a provider's claims containing any selected itemset and
   the number of distinct selected itemsets billed.

Leakage controls
----------------
* Itemsets and rules are learned from training providers' claims and labels only. Validation and
  test providers are transformed with the frozen rule set.
* Features for *training* providers are computed **out-of-fold**: rules are re-selected without
  the provider's own fold (a label-independent shuffled KFold), so a provider's label has no effect
  on its own feature - verified exactly in the test-suite.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules
from scipy import sparse
from sklearn.model_selection import KFold

from mediguard.data.ingest import DIAG_COLS, PROC_COLS

log = logging.getLogger(__name__)

APRIORI_FEATURES = ["suspicious_itemset_claims", "suspicious_itemset_variety"]


def claim_items(claims: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Long format (claim_row, item) pairs, de-duplicated per claim."""
    rows, items = [], []
    for prefix, cols in (("D_", DIAG_COLS), ("P_", PROC_COLS)):
        stacked = claims[cols].reset_index(drop=True).stack()
        rows.append(stacked.index.get_level_values(0).to_numpy())
        items.append((prefix + stacked.astype(str)).to_numpy())
    long = pd.DataFrame({"row": np.concatenate(rows), "item": np.concatenate(items)}).drop_duplicates()
    return long["row"].to_numpy(), long["item"].to_numpy()


def item_matrix(claims: pd.DataFrame, vocab: list[str]) -> sparse.csc_matrix:
    rows, items = claim_items(claims)
    index = {it: i for i, it in enumerate(vocab)}
    cols = pd.Series(items).map(index)
    keep = cols.notna().to_numpy()
    data = np.ones(keep.sum(), dtype=bool)
    return sparse.csr_matrix((data, (rows[keep], cols[keep].to_numpy(dtype=int))),
                             shape=(len(claims), len(vocab)), dtype=bool).tocsc()


def apriori_sparse(X: sparse.csc_matrix, min_support: float, max_len: int) -> list[tuple[tuple[int, ...], int]]:
    """Apriori over a boolean transactions x items matrix.

    Level-wise search: candidates of size k are joined from frequent (k-1)-itemsets sharing a
    (k-2)-prefix and pruned unless every (k-1)-subset is frequent (downward closure). Supports of
    pairs come from a single sparse product X^T X; larger candidates are counted by intersecting
    the row sets of their items. Returns ``[(itemset as sorted item indices, support count), ...]``.
    Equivalent to ``mlxtend.frequent_patterns.apriori`` (checked in the test-suite) but avoids
    densifying a ~450k x 300 matrix.
    """
    n = X.shape[0]
    min_count = int(np.ceil(min_support * n))
    X = X.tocsc().astype(np.int32)
    single = np.asarray(X.sum(axis=0)).ravel()
    level = {(i,): int(c) for i, c in enumerate(single) if c >= min_count}
    found = list(level.items())
    if max_len >= 2 and level:
        items = np.array([i for (i,) in level])
        sub = X[:, items]
        gram = (sub.T @ sub).toarray()
        level = {}
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                if gram[a, b] >= min_count:
                    level[(int(items[a]), int(items[b]))] = int(gram[a, b])
        found += list(level.items())
    rows = {i: X.indices[X.indptr[i]:X.indptr[i + 1]] for i in range(X.shape[1])}
    k = 3
    while k <= max_len and level:
        prev = sorted(level)
        prev_set = set(prev)
        candidates = []
        for a in range(len(prev)):
            for b in range(a + 1, len(prev)):
                if prev[a][:-1] != prev[b][:-1]:
                    break
                cand = prev[a] + (prev[b][-1],)
                if all(cand[:j] + cand[j + 1:] in prev_set for j in range(k)):
                    candidates.append(cand)
        level = {}
        for cand in candidates:
            r = rows[cand[0]]
            for it in cand[1:]:
                r = np.intersect1d(r, rows[it], assume_unique=True)
            if len(r) >= min_count:
                level[cand] = int(len(r))
        found += list(level.items())
        k += 1
    return found


def itemset_membership(X: sparse.csc_matrix, itemsets: list[tuple[int, ...]]) -> sparse.csc_matrix:
    """Claims x itemsets boolean matrix: claim contains every item of the itemset."""
    rows_all, cols_all = [], []
    for j, its in enumerate(itemsets):
        rows = X.indices[X.indptr[its[0]]:X.indptr[its[0] + 1]]
        for it in its[1:]:
            rows = np.intersect1d(rows, X.indices[X.indptr[it]:X.indptr[it + 1]], assume_unique=True)
        rows_all.append(rows)
        cols_all.append(np.full(len(rows), j))
    r = np.concatenate(rows_all) if rows_all else np.array([], dtype=int)
    c = np.concatenate(cols_all) if cols_all else np.array([], dtype=int)
    return sparse.csr_matrix((np.ones(len(r), dtype=bool), (r, c)), shape=(X.shape[0], len(itemsets))).tocsc()


class CodeAssociationMiner:
    def __init__(self, min_support=0.0015, max_len=3, min_claims=50, min_providers=15,
                 min_fraud_providers=5, min_lift=1.3, top_k=40, inner_folds=3, random_state=42):
        self.min_support = min_support
        self.max_len = max_len
        self.min_claims = min_claims
        self.min_providers = min_providers
        self.min_fraud_providers = min_fraud_providers
        self.min_lift = min_lift
        self.top_k = top_k
        self.inner_folds = inner_folds
        self.random_state = random_state

    # -- mining ---------------------------------------------------------------------------------
    def _mine(self, claims: pd.DataFrame) -> None:
        rows, items = claim_items(claims)
        n = len(claims)
        support = pd.Series(rows).groupby(items).nunique() / n
        self.vocab_ = sorted(support[support >= self.min_support].index)
        X = item_matrix(claims, self.vocab_)
        found = apriori_sparse(X, self.min_support, self.max_len)
        freq = pd.DataFrame({
            "support": [cnt / n for _, cnt in found],
            "itemsets": [frozenset(self.vocab_[i] for i in its) for its, _ in found],
        })
        self.n_transactions_ = n
        self.n_frequent_itemsets_ = int(len(freq))
        rules = association_rules(freq, num_itemsets=n, metric="lift", min_threshold=1.0)
        rules = rules.sort_values(["lift", "support"], ascending=False).head(30)
        self.code_rules_ = [{
            "antecedent": sorted(r.antecedents), "consequent": sorted(r.consequents),
            "support": float(r.support), "confidence": float(r.confidence), "lift": float(r.lift),
        } for r in rules.itertuples()]
        pos = {it: i for i, it in enumerate(self.vocab_)}
        multi = freq[freq["itemsets"].map(len) >= 2]
        self.itemsets_ = [tuple(sorted(pos[i] for i in s)) for s in multi["itemsets"]]
        self._X_train = X

    # -- rule scoring on a labelled subset of providers -------------------------------------------
    def _score(self, counts: sparse.csr_matrix, totals: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        n_claims = totals.sum()
        fraud_claims = totals[y == 1].sum()
        base = fraud_claims / max(n_claims, 1)
        itemset_claims = np.asarray(counts.sum(axis=0)).ravel()
        fraud_itemset_claims = np.asarray(counts[y == 1].sum(axis=0)).ravel()
        present = counts > 0
        providers = np.asarray(present.sum(axis=0)).ravel()
        fraud_providers = np.asarray(present[y == 1].sum(axis=0)).ravel()
        conf = np.divide(fraud_itemset_claims, itemset_claims, out=np.zeros(len(itemset_claims)),
                         where=itemset_claims > 0)
        return pd.DataFrame({
            "itemset_claims": itemset_claims, "providers": providers, "fraud_providers": fraud_providers,
            "support": fraud_itemset_claims / max(n_claims, 1), "confidence": conf,
            "lift": conf / base if base > 0 else 0.0,
        })

    def _select(self, scored: pd.DataFrame) -> np.ndarray:
        ok = ((scored["itemset_claims"] >= self.min_claims) & (scored["providers"] >= self.min_providers)
              & (scored["fraud_providers"] >= self.min_fraud_providers) & (scored["lift"] >= self.min_lift))
        return scored[ok].sort_values("lift", ascending=False).head(self.top_k).index.to_numpy()

    @staticmethod
    def _provider_features(member: sparse.csc_matrix, claim_provider: np.ndarray, providers: np.ndarray,
                           selected: np.ndarray) -> pd.DataFrame:
        """Aggregate claim-level rule hits for ``providers`` (claims of other providers are ignored)."""
        idx = pd.Index(providers).get_indexer(claim_provider)
        keep = np.flatnonzero(idx >= 0)
        P = sparse.csr_matrix((np.ones(len(keep)), (idx[keep], keep)), shape=(len(providers), member.shape[0]))
        if len(selected):
            sub = member[:, selected].astype(float)
            flagged = (np.asarray(sub.sum(axis=1)).ravel() > 0).astype(float)
            claims_flagged = P @ flagged
            variety = np.asarray(((P @ sub) > 0).sum(axis=1)).ravel().astype(float)
        else:
            claims_flagged = np.zeros(len(providers))
            variety = np.zeros(len(providers))
        return pd.DataFrame({"suspicious_itemset_claims": claims_flagged,
                             "suspicious_itemset_variety": variety}, index=providers)

    # -- public API -----------------------------------------------------------------------------
    def fit_transform(self, train_claims: pd.DataFrame, train_labels: pd.Series) -> pd.DataFrame:
        """Mine on training claims and return **out-of-fold** features for training providers.

        ``train_labels`` is indexed by Provider (1 = fraud-labelled).
        """
        self._mine(train_claims)
        member = itemset_membership(self._X_train, self.itemsets_)
        claim_provider = train_claims["Provider"].to_numpy()
        providers = np.array(sorted(pd.unique(claim_provider)))
        p_index = pd.Series(np.arange(len(providers)), index=providers)
        P = sparse.csr_matrix((np.ones(len(claim_provider)), (p_index[claim_provider].to_numpy(),
                               np.arange(len(claim_provider)))), shape=(len(providers), len(claim_provider)))
        counts = (P @ member.astype(float)).tocsr()
        totals = np.asarray(P.sum(axis=1)).ravel()
        y = train_labels.reindex(providers).to_numpy(dtype=int)

        # Final rule set from all training providers (used for every unseen provider).
        self.rules_ = self._score(counts, totals, y)
        self.selected_ = self._select(self.rules_)
        self.selected_itemsets_ = [self.itemsets_[i] for i in self.selected_]

        # Out-of-fold features for the training providers themselves.
        oof = pd.DataFrame(index=providers, columns=APRIORI_FEATURES, dtype=float)
        # Label-independent partition: a provider's own label cannot influence which providers'
        # labels are used to build its features (a stratified split would make it depend on it).
        kf = KFold(n_splits=self.inner_folds, shuffle=True, random_state=self.random_state)
        self.inner_rule_counts_ = []
        for fit_idx, hold_idx in kf.split(providers):
            sel = self._select(self._score(counts[fit_idx], totals[fit_idx], y[fit_idx]))
            self.inner_rule_counts_.append(int(len(sel)))
            feats = self._provider_features(member, claim_provider, providers[hold_idx], sel)
            oof.loc[providers[hold_idx]] = feats.to_numpy()
        self.train_providers_ = set(providers)
        del self._X_train
        return oof

    def transform(self, claims: pd.DataFrame) -> pd.DataFrame:
        X = item_matrix(claims, self.vocab_)
        member = itemset_membership(X, self.selected_itemsets_)
        claim_provider = claims["Provider"].to_numpy()
        providers = np.array(sorted(pd.unique(claim_provider)))
        return self._provider_features(member, claim_provider, providers, np.arange(len(self.selected_itemsets_)))

    def provider_rule_hits(self, claims: pd.DataFrame, top_n: int = 10) -> dict[str, list[dict]]:
        """For each provider: the selected itemsets it billed, with claim counts and rule lift."""
        member = itemset_membership(item_matrix(claims, self.vocab_), self.selected_itemsets_)
        prov = claims["Provider"].to_numpy()
        providers = np.array(sorted(pd.unique(prov)))
        idx = pd.Index(providers).get_indexer(prov)
        P = sparse.csr_matrix((np.ones(len(prov)), (idx, np.arange(len(prov)))), shape=(len(providers), len(prov)))
        counts = (P @ member.astype(float)).tocsr()
        names = [" + ".join(self.vocab_[i] for i in its) for its in self.selected_itemsets_]
        lifts = self.rules_.loc[self.selected_, "lift"].to_numpy()
        out = {}
        for r, p in enumerate(providers):
            row = counts.getrow(r)
            hits = [{"itemset": names[j], "claims": int(v), "lift": round(float(lifts[j]), 3)}
                    for j, v in zip(row.indices, row.data)]
            out[str(p)] = sorted(hits, key=lambda h: -h["claims"])[:top_n]
        return out

    def rule_report(self) -> dict:
        rules = self.rules_.loc[self.selected_].copy()
        rules["itemset"] = [[self.vocab_[i] for i in self.itemsets_[j]] for j in self.selected_]
        return {
            "n_transactions": self.n_transactions_,
            "n_items": len(self.vocab_),
            "n_frequent_itemsets": self.n_frequent_itemsets_,
            "n_candidate_combinations": len(self.itemsets_),
            "n_selected_rules": int(len(self.selected_)),
            "inner_fold_rule_counts": self.inner_rule_counts_,
            "params": {k: getattr(self, k) for k in ("min_support", "max_len", "min_claims", "min_providers",
                                                     "min_fraud_providers", "min_lift", "top_k")},
            "fraud_rules": [{
                "itemset": r.itemset, "claims": int(r.itemset_claims), "providers": int(r.providers),
                "fraud_labelled_providers": int(r.fraud_providers), "support": float(r.support),
                "confidence": float(r.confidence), "lift": float(r.lift),
            } for r in rules.itertuples()],
            "code_rules": self.code_rules_,
        }
