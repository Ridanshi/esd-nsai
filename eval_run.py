from src.data.loader import load_dataset
from src.models.model_a import run_model_a
from src.models.model_b import run_model_b
from src.models.model_c import run_model_c
from src.models.model_ensemble import run_model_ensemble
from src.evaluation.metrics import print_comparison_table, run_mcnemar_test

X_clinical, X_histopath, X_all, y = load_dataset()
res_a = run_model_a(X_all, y)
res_b = run_model_b(X_clinical, y)
res_c = run_model_c(X_clinical, y)
res_d = run_model_ensemble(X_clinical, y)

print_comparison_table(res_a, res_b, res_c, res_d)
print()

r_bc = run_mcnemar_test(res_b, res_c)
print("McNemar Test (B vs C):")
print(f"  a={r_bc['a']}  b={r_bc['b']}  c={r_bc['c']}  d={r_bc['d']}")
print(f"  chi2={r_bc['chi2_stat']}  p={r_bc['p_value']}  significant={r_bc['significant']}")
print(f"  {r_bc['interpretation']}")
print()

r_cd = run_mcnemar_test(res_c, res_d)
print("McNemar Test (C vs D/Ensemble):")
print(f"  a={r_cd['a']}  b(C wins)={r_cd['b']}  c(D wins)={r_cd['c']}  d={r_cd['d']}")
print(f"  chi2={r_cd['chi2_stat']}  p={r_cd['p_value']}  significant={r_cd['significant']}")
print(f"  {r_cd['interpretation']}")
