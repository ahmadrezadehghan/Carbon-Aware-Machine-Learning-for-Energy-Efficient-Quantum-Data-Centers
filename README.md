# مرکز دادهٔ کوانتومی سبز (GQDC) — اسکلت پروژه

این مخزن برای پژوهش «زمان‌بندی و کنترل آگاه از انرژی/کربن برای بارهای کوانتومی» آماده شده است.
هدف: **کاهش ≥۲۰٪ انرژی به‌ازای هر Job** بدون افزایش معنی‌دار نقض SLA، با کد قابل تکرار.

## اجرا (حداقل)
```bash
python -m venv .venv && source .venv/bin/activate    # ویندوز: .venv\Scripts\activate
pip install -r requirements-min.txt
python -m gqdc.experiments.run_ablation --config configs/config.yaml
```

## اجرا (کامل با MILP/RL)
```bash
pip install -r requirements.txt
# MILP نمونه (پس از تکمیل فایل milp.py)
# python -m gqdc.experiments.evaluate --scheduler milp --config configs/config.yaml

# آموزش PPO (پس از تکمیل/بهبود env و train)
# python -m gqdc.rl.train_ppo --config configs/config.yaml
```

### ساختار
- `gqdc/` کد بسته (package) شامل زیرماژول‌ها: simulate، quantum، scheduler، rl، control، metrics، experiments
- `configs/` تنظیمات YAML
- `docs/` اسناد فارسی (شامل الگوی North Star)
- `requirements*.txt` وابستگی‌ها
- `LICENSE` و `.gitignore`

---
تاریخ تولید: 2025-10-26 22:41
