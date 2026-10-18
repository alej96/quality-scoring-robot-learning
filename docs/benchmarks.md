# demoqc community benchmark

Generated 2026-10-18 with demoqc v0.1.0 (capped at 200 episodes/dataset). Regenerate with `python scripts/benchmark.py`; this also doubles as a regression check for the heuristics — a diff here after a threshold change shows exactly which real datasets moved, and by how much.

| Dataset | Episodes | Frames | Mean score | Median score | Trimmable idle | Dup. groups | Top flags |
|---|---|---|---|---|---|---|---|
| lerobot/svla_so101_pickplace | 50 | 11939 | 97.1 | 97.3 | 71.6s | 0 | idle x47 |
| lerobot/pusht | 200 | 24892 | 100.0 | 100.0 | 0.0s | 0 | none |
| masato-ka/so100_cutlery_handling_simple | 50 | 29853 | 92.9 | 94.1 | 181.6s | 0 | idle x52, length x2 |
| aaronsu11/so100_lego | 100 | 55617 | 92.2 | 91.5 | 445.1s | 0 | idle x119, smoothness x1 |
| shylee/so100_cube | 200 | 58213 | 99.2 | 100.0 | 16.6s | 0 | idle x14, smoothness x8, length x1 |
| jccj/so100_block_in_cup | 50 | 21782 | 95.2 | 95.7 | 104.7s | 0 | idle x47, smoothness x2 |
