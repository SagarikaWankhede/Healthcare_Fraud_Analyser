import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter
from typing import List, Tuple, Set, Dict

DIAG_COLS = [f'ClmDiagnosisCode_{i}' for i in range(1, 11)]
PROC_COLS = [f'ClmProcedureCode_{i}' for i in range(1, 7)]

class SuspiciousCodeMiner:
    """
    Mines frequent diagnosis and procedure code combinations using Apriori association principles.
    Identifies code pairs that appear disproportionately more frequently in claims filed
    by fraudulent providers compared to legitimate providers.
    
    Guarantees zero data leakage by extracting rules strictly within the training partition.
    """
    def __init__(self, min_fraud_support: float = 0.002, min_enrichment_ratio: float = 1.8, top_k_rules: int = 25):
        self.min_fraud_support = min_fraud_support
        self.min_enrichment_ratio = min_enrichment_ratio
        self.top_k_rules = top_k_rules
        self.suspicious_pairs_: Set[Tuple[str, str]] = set()
        self.rule_stats_: List[Dict] = []

    def _extract_claim_baskets(self, claims_df: pd.DataFrame) -> List[Tuple[str, List[str]]]:
        """Extracts sorted unique diagnosis and procedure codes for each claim."""
        baskets = []
        for row in claims_df[['Provider'] + DIAG_COLS + PROC_COLS].itertuples(index=False):
            provider = row[0]
            items = []
            # Diag codes: row[1..10]
            for val in row[1:11]:
                if pd.notna(val) and str(val).strip() != '':
                    items.append(f"D_{str(val).strip()}")
            # Proc codes: row[11..16]
            for val in row[11:17]:
                if pd.notna(val) and str(val).strip() != '':
                    items.append(f"P_{str(val).strip()}")
            
            unique_items = sorted(list(set(items)))
            baskets.append((provider, unique_items))
        return baskets

    def fit(self, train_claims: pd.DataFrame, train_labels: pd.DataFrame):
        """
        Mines suspicious code combinations strictly using training claims and provider labels.
        """
        print(f"Mining suspicious diagnosis/procedure code combinations on {len(train_claims):,} training claims...")
        # Map fraud labels to claims
        label_map = dict(zip(train_labels['Provider'], train_labels['PotentialFraud']))
        train_claims = train_claims.copy()
        train_claims['PotentialFraud'] = train_claims['Provider'].map(label_map).fillna(0).astype(int)

        fraud_claims = train_claims[train_claims['PotentialFraud'] == 1]
        legit_claims = train_claims[train_claims['PotentialFraud'] == 0]

        n_fraud = len(fraud_claims)
        n_legit = len(legit_claims)
        print(f"Training partition: {n_fraud:,} fraud claims vs {n_legit:,} legitimate claims.")

        fraud_baskets = self._extract_claim_baskets(fraud_claims)
        legit_baskets = self._extract_claim_baskets(legit_claims)

        # Count pair occurrences in fraud claims
        fraud_pair_counts = Counter()
        for _, items in fraud_baskets:
            if len(items) >= 2:
                for pair in combinations(items, 2):
                    fraud_pair_counts[pair] += 1

        # Count pair occurrences in legitimate claims
        legit_pair_counts = Counter()
        for _, items in legit_baskets:
            if len(items) >= 2:
                for pair in combinations(items, 2):
                    legit_pair_counts[pair] += 1

        min_count = int(self.min_fraud_support * n_fraud)
        rules = []

        for pair, f_count in fraud_pair_counts.items():
            if f_count >= min_count:
                f_supp = f_count / n_fraud
                l_count = legit_pair_counts.get(pair, 0)
                l_supp = l_count / n_legit
                
                # Enrichment / Lift ratio
                enrichment = (f_supp + 1e-6) / (l_supp + 1e-6)
                if enrichment >= self.min_enrichment_ratio:
                    rules.append({
                        'pair': pair,
                        'item_1': pair[0],
                        'item_2': pair[1],
                        'fraud_support': f_supp,
                        'legit_support': l_supp,
                        'enrichment_ratio': enrichment,
                        'fraud_count': f_count,
                        'legit_count': l_count
                    })

        # Sort by enrichment ratio descending
        rules.sort(key=lambda x: x['enrichment_ratio'], reverse=True)
        top_rules = rules[:self.top_k_rules]
        self.rule_stats_ = top_rules
        self.suspicious_pairs_ = {r['pair'] for r in top_rules}

        print(f"Discovered {len(self.suspicious_pairs_)} suspicious code combinations (Lift >= {self.min_enrichment_ratio:.1f}x).")
        if top_rules:
            top_rule = top_rules[0]
            print(f"Top Rule: {top_rule['item_1']} + {top_rule['item_2']} (Enrichment: {top_rule['enrichment_ratio']:.2f}x, Fraud Supp: {top_rule['fraud_support']:.3%})")
        return self

    def transform(self, claims_df: pd.DataFrame, provider_list: List[str]) -> pd.DataFrame:
        """
        Maps mined suspicious combinations back onto provider claims to generate provider-level features.
        """
        baskets = self._extract_claim_baskets(claims_df)
        provider_flagged_claims = Counter()
        provider_distinct_combos = {p: set() for p in provider_list}
        provider_total_claims = Counter()

        for provider, items in baskets:
            provider_total_claims[provider] += 1
            if len(items) >= 2 and len(self.suspicious_pairs_) > 0:
                item_set = set(items)
                has_suspicious = False
                for pair in self.suspicious_pairs_:
                    if pair[0] in item_set and pair[1] in item_set:
                        has_suspicious = True
                        if provider in provider_distinct_combos:
                            provider_distinct_combos[provider].add(pair)
                if has_suspicious:
                    provider_flagged_claims[provider] += 1

        records = []
        for p in provider_list:
            total = provider_total_claims.get(p, 0)
            flagged = provider_flagged_claims.get(p, 0)
            distinct_combos = len(provider_distinct_combos.get(p, set()))
            ratio = flagged / max(total, 1)
            records.append({
                'Provider': p,
                'suspicious_code_claim_count': flagged,
                'suspicious_code_claim_ratio': ratio,
                'suspicious_combo_variety': distinct_combos
            })

        return pd.DataFrame(records)
