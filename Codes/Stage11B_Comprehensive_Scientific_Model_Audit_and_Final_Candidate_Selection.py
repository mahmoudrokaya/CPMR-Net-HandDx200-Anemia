from pathlib import Path
import json
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================
# STAGE 11B — COMPREHENSIVE SCIENTIFIC MODEL AUDIT
# ============================================================
# Purpose:
#   Convert Stage 11A's unified comparison into a publication-ready
#   scientific model audit and final candidate selection framework.
#
# Important:
#   This stage does NOT train, tune, or redesign any model.
#   It audits evidence quality, generalization, interpretability,
#   computational feasibility, reproducibility, and scientific value.
# ============================================================

BASE_DIR = Path(r'D:\47\472\New-Papers\Anemia_Paper')
OUTPUTS_DIR = BASE_DIR / 'Outputs'

STAGE11A_DIR = OUTPUTS_DIR / 'Stage11A_Unified_Model_Comparison_and_Decision_Audit'
STAGE11A_TABLES = STAGE11A_DIR / 'tables'

OUTPUT_DIR = OUTPUTS_DIR / 'Stage11B_Comprehensive_Scientific_Model_Audit_and_Final_Candidate_Selection'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'
REPORTS_DIR = OUTPUT_DIR / 'reports'
SUMMARY_DIR = OUTPUT_DIR / 'summary'

for d in [OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR, SUMMARY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# INPUT FILES
# ============================================================

STAGE11A_FULL = STAGE11A_TABLES / 'stage11a_unified_model_comparison_full.csv'
STAGE11A_RANKING = STAGE11A_TABLES / 'stage11a_final_decision_ranking.csv'

# ============================================================
# HELPERS
# ============================================================

def read_csv_safe(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def as_float(x):
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, str) and '±' in x:
            return float(x.split('±')[0].strip())
        return float(x)
    except Exception:
        return np.nan


def level_from_numeric(value, high=0.75, moderate=0.65):
    value = as_float(value)
    if pd.isna(value):
        return 'Not available'
    if value >= high:
        return 'Strong'
    if value >= moderate:
        return 'Moderate'
    return 'Weak'


def evidence_strength(row):
    tier = str(row.get('evidence_tier', ''))
    gap = as_float(row.get('generalization_gap_val_minus_holdout', np.nan))
    cv = as_float(row.get('repeated_cv_roc_auc_mean', np.nan))
    holdout = as_float(row.get('holdout_roc_auc', np.nan))
    val = as_float(row.get('validation_roc_auc', np.nan))

    if 'Tier 1' in tier and not pd.isna(cv):
        if cv >= 0.70:
            return 'Strong'
        if cv >= 0.60:
            return 'Moderate'
        return 'Weak'
    if 'Tier 2' in tier and not pd.isna(holdout):
        if not pd.isna(gap) and gap > 0.15:
            return 'Weak'
        if holdout >= 0.70:
            return 'Strong'
        if holdout >= 0.60:
            return 'Moderate'
        return 'Weak'
    if 'Tier 3' in tier and not pd.isna(val):
        if val >= 0.78:
            return 'Moderate'
        return 'Weak'
    return 'Not available'


def generalization_strength(row):
    gap = as_float(row.get('generalization_gap_val_minus_holdout', np.nan))
    cv = as_float(row.get('repeated_cv_roc_auc_mean', np.nan))
    cv_std = as_float(row.get('repeated_cv_roc_auc_std', np.nan))
    holdout = as_float(row.get('holdout_roc_auc', np.nan))

    if not pd.isna(cv):
        if cv >= 0.70 and (pd.isna(cv_std) or cv_std <= 0.08):
            return 'Strong'
        if cv >= 0.60:
            return 'Moderate'
        return 'Weak'
    if not pd.isna(holdout):
        if not pd.isna(gap) and gap <= 0.10 and holdout >= 0.70:
            return 'Strong'
        if not pd.isna(gap) and gap <= 0.15 and holdout >= 0.60:
            return 'Moderate'
        return 'Weak'
    return 'Not established'


def computational_strength(row):
    stage = str(row.get('stage', ''))
    family = str(row.get('model_family', ''))
    if 'Classical' in family or stage == 'Stage6D6B':
        return 'Strong'
    if stage in {'Stage10X', 'Stage10Z', 'Stage10Y'}:
        return 'Moderate'
    if stage == 'Stage10L':
        return 'Moderate'
    return 'Not available'


def interpretability_strength(row):
    stage = str(row.get('stage', ''))
    representation = str(row.get('representation_learning', ''))
    family = str(row.get('model_family', ''))
    if stage == 'Stage6D6B':
        return 'Strong'
    if 'CPMR' in family or stage.startswith('Stage10'):
        if stage in {'Stage10X', 'Stage10Z', 'Stage10Y'}:
            return 'Strong'
        return 'Moderate'
    if 'Handcrafted' in representation:
        return 'Strong'
    return 'Moderate'


def novelty_strength(row):
    stage = str(row.get('stage', ''))
    family = str(row.get('model_family', ''))
    if stage == 'Stage6D6B':
        return 'Moderate'
    if 'CPMR' in family or stage.startswith('Stage10'):
        if stage in {'Stage10X', 'Stage10Z'}:
            return 'Strong'
        return 'Moderate'
    return 'Moderate'


def reproducibility_strength(row):
    files = as_float(row.get('evidence_files_found', np.nan))
    found = row.get('stage_dir_found', False)
    if isinstance(found, str):
        found = found.lower() == 'true'
    if found and not pd.isna(files) and files >= 3:
        return 'Strong'
    if found:
        return 'Moderate'
    return 'Weak'


def score_level(level):
    return {'Strong': 3, 'Moderate': 2, 'Weak': 1, 'Not established': 0.5, 'Not available': 0}.get(level, 0)


def make_bar(df, value_col, label_col, title, output_file):
    if df.empty or value_col not in df.columns:
        return
    plot_df = df.copy().sort_values(value_col, ascending=True)
    labels = plot_df[label_col].astype(str).tolist()
    values = plot_df[value_col].astype(float).tolist()
    plt.figure(figsize=(12, max(4, 0.7 * len(plot_df))))
    plt.barh(labels, values)
    plt.xlabel(value_col.replace('_', ' ').title())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

# ============================================================
# LOAD STAGE 11A OUTPUTS
# ============================================================

model_df = read_csv_safe(STAGE11A_FULL)
if model_df.empty:
    model_df = read_csv_safe(STAGE11A_RANKING)

if model_df.empty:
    raise FileNotFoundError(
        'Stage 11A comparison files were not found. Run Stage11A first or verify output paths.'
    )

# Normalize numeric columns
numeric_cols = [
    'validation_roc_auc', 'holdout_roc_auc', 'repeated_cv_roc_auc_mean',
    'repeated_cv_roc_auc_std', 'best_available_auc_conservative',
    'generalization_gap_val_minus_holdout', 'accuracy', 'precision',
    'recall', 'f1_score', 'pr_auc', 'balanced_accuracy', 'decision_score',
    'evidence_files_found'
]
for col in numeric_cols:
    if col in model_df.columns:
        model_df[col] = model_df[col].apply(as_float)

# Ensure display label
if 'audit_label' not in model_df.columns:
    model_df['audit_label'] = model_df['stage'].astype(str) + ' | ' + model_df['model_name'].astype(str)

# ============================================================
# COMPREHENSIVE AUDIT MATRIX
# ============================================================

records = []
for _, row in model_df.iterrows():
    rec = row.to_dict()

    rec['validation_performance_level'] = level_from_numeric(row.get('validation_roc_auc', np.nan), 0.78, 0.70)
    rec['holdout_performance_level'] = level_from_numeric(row.get('holdout_roc_auc', np.nan), 0.70, 0.60)
    rec['repeated_cv_performance_level'] = level_from_numeric(row.get('repeated_cv_roc_auc_mean', np.nan), 0.70, 0.60)
    rec['evidence_strength_level'] = evidence_strength(row)
    rec['generalization_strength_level'] = generalization_strength(row)
    rec['computational_feasibility_level'] = computational_strength(row)
    rec['interpretability_level'] = interpretability_strength(row)
    rec['reproducibility_level'] = reproducibility_strength(row)
    rec['scientific_novelty_level'] = novelty_strength(row)

    # Publication-suitability score avoids arbitrary performance-only ranking.
    # It combines qualitative levels with conservative AUC evidence.
    qualitative_levels = [
        rec['evidence_strength_level'],
        rec['generalization_strength_level'],
        rec['computational_feasibility_level'],
        rec['interpretability_level'],
        rec['reproducibility_level'],
        rec['scientific_novelty_level'],
    ]
    qualitative_score = np.mean([score_level(x) for x in qualitative_levels]) / 3.0
    auc_component = as_float(row.get('best_available_auc_conservative', np.nan))
    if pd.isna(auc_component):
        auc_component = 0.0

    # Penalize models with only validation evidence because final selection requires generalization evidence.
    tier = str(row.get('evidence_tier', ''))
    evidence_penalty = 0.00
    if 'Tier 3' in tier:
        evidence_penalty = 0.08
    elif 'Tier 2' in tier:
        gap = as_float(row.get('generalization_gap_val_minus_holdout', np.nan))
        if not pd.isna(gap) and gap > 0.15:
            evidence_penalty = 0.12

    rec['publication_suitability_score'] = round(
        max(0.0, 0.55 * qualitative_score + 0.45 * auc_component - evidence_penalty), 4
    )

    # Role assignment
    stage = str(row.get('stage', ''))
    if stage == 'Stage6D6B':
        rec['recommended_role'] = 'Mandatory validated handcrafted benchmark'
    elif stage == 'Stage10X':
        rec['recommended_role'] = 'Leading CPMR-Net final candidate pending generalization confirmation'
    elif stage == 'Stage10Z':
        rec['recommended_role'] = 'Secondary CPMR-Net candidate and evidence for teacher-guided learning'
    elif stage == 'Stage10Y':
        rec['recommended_role'] = 'Ablation evidence; not final candidate'
    elif stage == 'Stage10P':
        rec['recommended_role'] = 'Generalization diagnosis evidence; not final deployed model'
    elif stage == 'Stage10L':
        rec['recommended_role'] = 'Rejected as final candidate due to holdout generalization collapse'
    else:
        rec['recommended_role'] = 'Supporting evidence'

    records.append(rec)

audit_df = pd.DataFrame(records)

# Ranking: publication suitability, then conservative AUC
sort_cols = ['publication_suitability_score', 'best_available_auc_conservative']
audit_df = audit_df.sort_values(sort_cols, ascending=False).reset_index(drop=True)
audit_df.insert(0, 'stage11b_rank', np.arange(1, len(audit_df) + 1))

# ============================================================
# FINAL CANDIDATE SELECTION LOGIC
# ============================================================

benchmark_df = audit_df[audit_df['stage'].astype(str).eq('Stage6D6B')]
cpmr_df = audit_df[audit_df['stage'].astype(str).str.startswith('Stage10')].copy()

# Select leading CPMR candidate based on validation promise and audit role.
preferred = cpmr_df[cpmr_df['stage'].astype(str).eq('Stage10X')]
if not preferred.empty:
    final_cpmr = preferred.iloc[0].to_dict()
elif not cpmr_df.empty:
    final_cpmr = cpmr_df.iloc[0].to_dict()
else:
    final_cpmr = {}

benchmark = benchmark_df.iloc[0].to_dict() if not benchmark_df.empty else {}

selection_records = []
if benchmark:
    selection_records.append({
        'selection_category': 'Strongest validated benchmark',
        'stage': benchmark.get('stage'),
        'model_name': benchmark.get('model_name'),
        'decision': 'Retain as mandatory comparator, not replace by validation-only deep model.',
        'justification': 'It has the strongest conservative repeated-CV evidence and remains the safest validated benchmark.'
    })
if final_cpmr:
    selection_records.append({
        'selection_category': 'Final CPMR-Net candidate',
        'stage': final_cpmr.get('stage'),
        'model_name': final_cpmr.get('model_name'),
        'decision': 'Select as leading CPMR-Net candidate, conditional on Stage 11C/11D generalization verification.',
        'justification': 'It provides the strongest stabilized CPMR-Net validation evidence while preserving the fixed CPMR-Net hierarchy.'
    })

selection_df = pd.DataFrame(selection_records)

# ============================================================
# CLAIM POLICY FOR PAPER
# ============================================================

claim_policy = [
    {
        'claim_type': 'Safe claim',
        'claim': 'The handcrafted RBFSVM benchmark remains the strongest validated comparator.',
        'support': 'Stage 6D6B repeated-CV ROC-AUC = 0.7447.'
    },
    {
        'claim_type': 'Safe claim',
        'claim': 'CPMR-Net is technically valid and scientifically structured, but deep-learning generalization remains unstable on 198 participants.',
        'support': 'Stage 10P repeated-CV ROC-AUC = 0.5923 ± 0.1022; Stage 10L holdout ROC-AUC = 0.5556.'
    },
    {
        'claim_type': 'Conditional claim',
        'claim': 'Progressive contrastive fine-tuning is the leading CPMR-Net candidate.',
        'support': 'Stage 10X validation ROC-AUC = 0.7879; final claim requires holdout/repeated-CV confirmation.'
    },
    {
        'claim_type': 'Avoided claim',
        'claim': 'CPMR-Net definitively outperforms the handcrafted benchmark.',
        'support': 'Not yet supported because Stage 10X currently has validation-only evidence.'
    },
    {
        'claim_type': 'Avoided claim',
        'claim': 'The initial full CPMR-Net is the best model because it achieved validation ROC-AUC 0.8052.',
        'support': 'Rejected because holdout ROC-AUC dropped to 0.5556.'
    }
]
claim_policy_df = pd.DataFrame(claim_policy)

# ============================================================
# SAVE TABLES
# ============================================================

audit_df.to_csv(TABLES_DIR / 'stage11b_comprehensive_scientific_audit_matrix.csv', index=False)
selection_df.to_csv(TABLES_DIR / 'stage11b_final_candidate_selection.csv', index=False)
claim_policy_df.to_csv(TABLES_DIR / 'stage11b_publication_claim_policy.csv', index=False)

level_cols = [
    'stage', 'model_name', 'validation_performance_level', 'holdout_performance_level',
    'repeated_cv_performance_level', 'evidence_strength_level', 'generalization_strength_level',
    'computational_feasibility_level', 'interpretability_level', 'reproducibility_level',
    'scientific_novelty_level', 'publication_suitability_score', 'recommended_role'
]
level_cols = [c for c in level_cols if c in audit_df.columns]
audit_df[level_cols].to_csv(TABLES_DIR / 'stage11b_qualitative_decision_matrix.csv', index=False)

# ============================================================
# FIGURES
# ============================================================

make_bar(
    audit_df,
    'publication_suitability_score',
    'audit_label',
    'Stage 11B publication-suitability ranking',
    FIGURES_DIR / 'stage11b_publication_suitability_ranking.png'
)

if 'best_available_auc_conservative' in audit_df.columns:
    make_bar(
        audit_df,
        'best_available_auc_conservative',
        'audit_label',
        'Stage 11B conservative ROC-AUC evidence',
        FIGURES_DIR / 'stage11b_conservative_auc_evidence.png'
    )

# Qualitative heatmap-like table encoded numerically
heat_cols = [
    'evidence_strength_level', 'generalization_strength_level', 'computational_feasibility_level',
    'interpretability_level', 'reproducibility_level', 'scientific_novelty_level'
]
heat_df = audit_df[['audit_label'] + heat_cols].copy()
heat_num = heat_df[heat_cols].applymap(score_level)
plt.figure(figsize=(12, max(4, 0.7 * len(heat_df))))
plt.imshow(heat_num.values, aspect='auto')
plt.yticks(range(len(heat_df)), heat_df['audit_label'])
plt.xticks(range(len(heat_cols)), [c.replace('_level', '').replace('_', ' ').title() for c in heat_cols], rotation=30, ha='right')
plt.colorbar(label='Qualitative score: 0=NA, 1=Weak, 2=Moderate, 3=Strong')
plt.title('Stage 11B qualitative scientific audit matrix')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'stage11b_qualitative_audit_matrix.png', dpi=300)
plt.close()

# ============================================================
# REPORT
# ============================================================

report_lines = []
report_lines.append('# Stage 11B — Comprehensive Scientific Model Audit and Final Candidate Selection')
report_lines.append('')
report_lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
report_lines.append('')
report_lines.append('## Purpose')
report_lines.append('')
report_lines.append('Stage 11B converts Stage 11A model comparison into a publication-oriented scientific audit. It does not train, tune, or redesign any model. The purpose is to determine which model should be treated as the validated benchmark, which CPMR-Net variant should be advanced as the final candidate, and which claims are scientifically safe for the manuscript.')
report_lines.append('')
report_lines.append('## Fixed Decision Rules')
report_lines.append('')
report_lines.append('- Participant-level diagnosis remains mandatory.')
report_lines.append('- The CPMR-Net hierarchy remains fixed.')
report_lines.append('- Validation-only results cannot justify final superiority claims.')
report_lines.append('- Repeated-CV and independent holdout evidence have priority over single validation split performance.')
report_lines.append('- Stage 6D6B remains the mandatory handcrafted benchmark until a CPMR-Net variant surpasses it under comparable generalization evidence.')
report_lines.append('')
report_lines.append('## Final Selection')
report_lines.append('')
if not selection_df.empty:
    report_lines.append(selection_df.to_markdown(index=False))
else:
    report_lines.append('No final selection could be made because required model records were unavailable.')
report_lines.append('')
report_lines.append('## Comprehensive Audit Ranking')
report_lines.append('')
display_cols = ['stage11b_rank', 'stage', 'model_name', 'best_available_auc_conservative', 'publication_suitability_score', 'evidence_strength_level', 'generalization_strength_level', 'recommended_role']
display_cols = [c for c in display_cols if c in audit_df.columns]
report_lines.append(audit_df[display_cols].to_markdown(index=False))
report_lines.append('')
report_lines.append('## Interpretation')
report_lines.append('')
report_lines.append('The audit confirms that Stage 6D6B is still the strongest validated benchmark because it has the strongest conservative repeated-CV evidence. Stage 10X is selected as the leading CPMR-Net candidate because it provides the strongest stabilized validation evidence among the advanced representation-learning variants. However, Stage 10X should not yet be claimed to definitively outperform Stage 6D6B until it receives comparable holdout or repeated-CV confirmation.')
report_lines.append('')
report_lines.append('Stage 10L is rejected as a final candidate despite its high validation ROC-AUC because the independent holdout result showed a substantial generalization collapse. Stage 10P remains scientifically important because it established that deep-learning generalization is unstable on the current 198-participant dataset.')
report_lines.append('')
report_lines.append('## Publication Claim Policy')
report_lines.append('')
report_lines.append(claim_policy_df.to_markdown(index=False))
report_lines.append('')
report_lines.append('## Recommended Next Stage')
report_lines.append('')
report_lines.append('Proceed to Stage 11C: conservative final-candidate verification of Stage 10X using holdout and/or repeated-CV evaluation, without changing the CPMR-Net architecture.')
report_lines.append('')
report_lines.append('## Generated Outputs')
report_lines.append('')
report_lines.append('- tables/stage11b_comprehensive_scientific_audit_matrix.csv')
report_lines.append('- tables/stage11b_qualitative_decision_matrix.csv')
report_lines.append('- tables/stage11b_final_candidate_selection.csv')
report_lines.append('- tables/stage11b_publication_claim_policy.csv')
report_lines.append('- figures/stage11b_publication_suitability_ranking.png')
report_lines.append('- figures/stage11b_conservative_auc_evidence.png')
report_lines.append('- figures/stage11b_qualitative_audit_matrix.png')

report_text = '\n'.join(report_lines)
(REPORTS_DIR / 'Stage11B_Comprehensive_Scientific_Model_Audit_and_Final_Candidate_Selection_Report.md').write_text(report_text, encoding='utf-8')

summary = {
    'stage': 'Stage11B',
    'models_audited': int(len(audit_df)),
    'top_publication_suitability_model': audit_df.iloc[0].to_dict() if not audit_df.empty else None,
    'validated_benchmark': benchmark,
    'final_cpmr_candidate': final_cpmr,
    'interpretation': 'Stage6D6B remains the validated benchmark; Stage10X is selected as the leading CPMR-Net candidate pending conservative generalization verification.',
    'recommended_next_stage': 'Stage11C: conservative final-candidate verification of Stage10X.',
    'output_dir': str(OUTPUT_DIR),
}
with open(SUMMARY_DIR / 'Stage11B_Comprehensive_Scientific_Model_Audit_and_Final_Candidate_Selection_Summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, default=str)

print('=' * 80)
print('STAGE 11B COMPREHENSIVE SCIENTIFIC MODEL AUDIT COMPLETED')
print('=' * 80)
print(f'Models audited: {len(audit_df)}')
if benchmark:
    print(f"Validated benchmark: {benchmark.get('stage')} | {benchmark.get('model_name')}")
if final_cpmr:
    print(f"Final CPMR-Net candidate: {final_cpmr.get('stage')} | {final_cpmr.get('model_name')}")
print('Top publication-suitability ranking:')
for _, r in audit_df.head(5).iterrows():
    print(f"  Rank {int(r['stage11b_rank'])}: {r['stage']} | score={r['publication_suitability_score']:.4f} | {r['recommended_role']}")
print(f'Results saved to: {OUTPUT_DIR}')
print('=' * 80)
