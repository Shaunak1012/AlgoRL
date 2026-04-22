# Ablation table (test split 2022–2024)

| label                           |   n_members | mode        |   ann_return |   sharpe |   sortino |   max_drawdown |   calmar |   n_days |
|:--------------------------------|------------:|:------------|-------------:|---------:|----------:|---------------:|---------:|---------:|
| A: MLP multi-asset, no CVaR     |           1 | single      |      -0.0319 |   0.1673 |    0.2390 |         0.5091 |  -0.0627 |      731 |
| B: + CVaR (λ=0.01)              |           1 | single      |      -0.0135 |   0.1799 |    0.2546 |         0.5139 |  -0.0262 |      731 |
| C: + Transformer, no early stop |           1 | single      |      -0.1084 |   0.0926 |    0.1491 |         0.7176 |  -0.1511 |      691 |
| D: + early stopping (50k)       |           1 | single      |       0.2902 |   0.7444 |    1.3955 |         0.5633 |   0.5151 |      691 |
| E: + 5-seed ensemble (mean)     |           5 | mean        |      -0.0880 |  -0.2024 |   -0.2771 |         0.4420 |  -0.1992 |      691 |
| F: + uncertainty sizing (FULL)  |           5 | uncertainty |       0.0127 |   0.2679 |    0.3752 |         0.0864 |   0.1468 |      691 |
