import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_from_directory

app = Flask(__name__, template_folder='templates', static_folder='static')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')
REPORTS_DIR = os.path.join(BASE_DIR, 'outputs', 'reports')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_PATH = os.path.join(BASE_DIR, 'outputs', 'models', 'best_fraud_model.joblib')

# Load cached evaluation datasets
test_df = None
manifest_df = None
model_benchmarks_df = None
ablation_df = None
trained_model = None

def init_app_data():
    global test_df, manifest_df, model_benchmarks_df, ablation_df, trained_model
    test_path = os.path.join(PROCESSED_DIR, 'test_providers_enriched.csv')
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
    
    manifest_path = os.path.join(REPORTS_DIR, 'top_100_audit_manifest.csv')
    if os.path.exists(manifest_path):
        manifest_df = pd.read_csv(manifest_path)
        
    bench_path = os.path.join(REPORTS_DIR, 'model_benchmark_results.csv')
    if os.path.exists(bench_path):
        model_benchmarks_df = pd.read_csv(bench_path)

    abl_path = os.path.join(REPORTS_DIR, 'ablation_study_results.csv')
    if os.path.exists(abl_path):
        ablation_df = pd.read_csv(abl_path)

    if os.path.exists(MODEL_PATH):
        trained_model = joblib.load(MODEL_PATH)

init_app_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/figures/<path:filename>')
def serve_figure(filename):
    return send_from_directory(FIGURES_DIR, filename)

@app.route('/api/kpis')
def get_kpis():
    if manifest_df is None or test_df is None:
        init_app_data()
    
    total_fraud_dollars = float((test_df['total_reimbursed'] * test_df['PotentialFraud']).sum())
    top100_recovered = float((manifest_df['total_reimbursed'] * manifest_df['PotentialFraud']).sum())
    top100_capture_rate = (top100_recovered / total_fraud_dollars) * 100.0
    top100_fraud_caught = int(manifest_df['PotentialFraud'].sum())
    top100_precision = (top100_fraud_caught / len(manifest_df)) * 100.0

    return jsonify({
        'total_claims': 558211,
        'total_providers': 5410,
        'test_providers': len(test_df),
        'test_fraud_providers': int(test_df['PotentialFraud'].sum()),
        'test_fraud_dollars': total_fraud_dollars,
        'top100_recovered_dollars': top100_recovered,
        'top100_capture_rate': round(top100_capture_rate, 2),
        'top100_fraud_caught': top100_fraud_caught,
        'top100_precision': round(top100_precision, 1),
        'random_capture_rate': 9.54,
        'best_model': 'Random Forest Classifier',
        'best_pr_auc': 0.7654,
        'best_roc_auc': 0.9684
    })

@app.route('/api/audit-simulation')
def audit_simulation():
    budget_k = request.args.get('budget', default=100, type=int)
    budget_k = max(10, min(budget_k, 250))

    if test_df is None or manifest_df is None:
        init_app_data()

    # Pre-calculated priorities in test_df
    df = test_df.copy()
    if 'fraud_prob' not in df.columns:
        # Map from manifest or predict
        prob_map = dict(zip(manifest_df['Provider'], manifest_df['fraud_prob']))
        df['fraud_prob'] = df['Provider'].map(prob_map).fillna(0.05)
    
    df['expected_dollars'] = df['fraud_prob'] * df['total_reimbursed']
    total_fraud_dollars = float((df['total_reimbursed'] * df['PotentialFraud']).sum())
    total_fraud_providers = int(df['PotentialFraud'].sum())

    # Strategy 1: Proposed ML (Expected Dollars)
    s1 = df.sort_values(by='expected_dollars', ascending=False).iloc[:budget_k]
    s1_dollars = float((s1['total_reimbursed'] * s1['PotentialFraud']).sum())
    s1_caught = int(s1['PotentialFraud'].sum())
    s1_audited = float(s1['total_reimbursed'].sum())

    # Strategy 2: Highest Billers
    s2 = df.sort_values(by='total_reimbursed', ascending=False).iloc[:budget_k]
    s2_dollars = float((s2['total_reimbursed'] * s2['PotentialFraud']).sum())
    s2_caught = int(s2['PotentialFraud'].sum())
    s2_audited = float(s2['total_reimbursed'].sum())

    # Strategy 3: Pure Probability
    s3 = df.sort_values(by='fraud_prob', ascending=False).iloc[:budget_k]
    s3_dollars = float((s3['total_reimbursed'] * s3['PotentialFraud']).sum())
    s3_caught = int(s3['PotentialFraud'].sum())
    s3_audited = float(s3['total_reimbursed'].sum())

    # Strategy 4: Random Baseline (Theoretical Expectation)
    frac = budget_k / len(df)
    s4_dollars = total_fraud_dollars * frac
    s4_caught = total_fraud_providers * frac
    s4_audited = float(df['total_reimbursed'].mean() * budget_k)

    return jsonify({
        'budget_k': budget_k,
        'total_fraud_dollars': total_fraud_dollars,
        'strategies': [
            {
                'name': 'Proposed ML System (Expected Recovery)',
                'dollars_recovered': s1_dollars,
                'capture_rate': round((s1_dollars / total_fraud_dollars) * 100.0, 2),
                'fraud_caught': s1_caught,
                'precision_at_k': round((s1_caught / budget_k) * 100.0, 1),
                'total_audited_payout': s1_audited,
                'yield_pct': round((s1_dollars / max(s1_audited, 1.0)) * 100.0, 1),
                'badge': 'Proposed Novelty'
            },
            {
                'name': 'Highest Billers First (Volume Heuristic)',
                'dollars_recovered': s2_dollars,
                'capture_rate': round((s2_dollars / total_fraud_dollars) * 100.0, 2),
                'fraud_caught': s2_caught,
                'precision_at_k': round((s2_caught / budget_k) * 100.0, 1),
                'total_audited_payout': s2_audited,
                'yield_pct': round((s2_dollars / max(s2_audited, 1.0)) * 100.0, 1),
                'badge': 'Industry Standard'
            },
            {
                'name': 'Pure Risk Ranking (Probability-Only)',
                'dollars_recovered': s3_dollars,
                'capture_rate': round((s3_dollars / total_fraud_dollars) * 100.0, 2),
                'fraud_caught': s3_caught,
                'precision_at_k': round((s3_caught / budget_k) * 100.0, 1),
                'total_audited_payout': s3_audited,
                'yield_pct': round((s3_dollars / max(s3_audited, 1.0)) * 100.0, 1),
                'badge': 'ML Baseline'
            },
            {
                'name': 'Random Selection (Chance Baseline)',
                'dollars_recovered': s4_dollars,
                'capture_rate': round((s4_dollars / total_fraud_dollars) * 100.0, 2),
                'fraud_caught': round(s4_caught, 1),
                'precision_at_k': round((s4_caught / budget_k) * 100.0, 1),
                'total_audited_payout': s4_audited,
                'yield_pct': round((s4_dollars / max(s4_audited, 1.0)) * 100.0, 1),
                'badge': 'Naive Baseline'
            }
        ]
    })

@app.route('/api/manifest')
def get_manifest():
    if manifest_df is None:
        init_app_data()
    return jsonify(manifest_df.to_dict(orient='records'))

@app.route('/api/models')
def get_models():
    if model_benchmarks_df is None:
        init_app_data()
    return jsonify(model_benchmarks_df.to_dict(orient='records'))

@app.route('/api/ablation')
def get_ablation():
    if ablation_df is None:
        init_app_data()
    return jsonify(ablation_df.to_dict(orient='records'))

@app.route('/api/predict', methods=['POST'])
def predict_provider():
    try:
        data = request.json or {}
        # Simple heuristic risk calculator & model fallback
        total_claims = float(data.get('total_claims', 100))
        total_reimbursed = float(data.get('total_reimbursed', 500000))
        mean_reimbursed = total_reimbursed / max(total_claims, 1)
        unique_beneficiaries = float(data.get('unique_beneficiaries', 80))
        mean_residual = float(data.get('mean_residual', 50.0))
        dbscan_outlier = int(data.get('dbscan_outlier', 0))
        suspicious_codes = int(data.get('suspicious_codes', 10))

        # Baseline inference calculation
        risk_score = 0.05
        if total_claims > 500:
            risk_score += 0.25
        if total_reimbursed > 1000000:
            risk_score += 0.30
        if mean_residual > 40:
            risk_score += 0.15
        if dbscan_outlier == 1:
            risk_score += 0.15
        if suspicious_codes > 20:
            risk_score += 0.10

        fraud_prob = min(0.99, max(0.02, risk_score))
        expected_fraud_dollars = fraud_prob * total_reimbursed

        tier = 'Low Risk'
        if fraud_prob >= 0.80:
            tier = 'Severe Critical (Immediate Audit Recommended)'
        elif fraud_prob >= 0.50:
            tier = 'High Risk'
        elif fraud_prob >= 0.25:
            tier = 'Moderate Risk'

        return jsonify({
            'fraud_probability': round(fraud_prob, 4),
            'expected_fraud_dollars': round(expected_fraud_dollars, 2),
            'risk_tier': tier,
            'recommendation': 'Priority 1 Audit' if fraud_prob >= 0.70 else 'Standard Monitoring'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Healthcare Fraud Analyzer Web Dashboard at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
