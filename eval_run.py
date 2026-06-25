from src.data.loader import load_dataset
from src.models.model_a import run_model_a
from src.models.model_b import run_model_b
from src.models.model_c import run_model_c
from src.evaluation.metrics import print_comparison_table, run_mcnemar_test

X_clinical, X_histopath, X_all, y = load_dataset()
res_a = run_model_a(X_all, y)
res_b = run_model_b(X_clinical, y)
res_c = run_model_c(X_clinical, y)

print_comparison_table(res_a, res_b, res_c)
print()

r = run_mcnemar_test(res_b, res_c)
print("McNemar Test (B vs C):")
print(f"  Both correct   (a): {r['a']}")
print(f"  B wins, C wrong(b): {r['b']}")
print(f"  C wins, B wrong(c): {r['c']}")
print(f"  Both wrong     (d): {r['d']}")
print(f"  chi2={r['chi2_stat']}  p={r['p_value']}  significant={r['significant']}")
print(f"  {r['interpretation']}")
