# Experiments

`quick_test.py` runs the current smoke experiment:

- direct cloud generation through `/generate`
- real SpecExtend through `/specextend/verify`
- `good` network simulation
- normalized result rows in `outputs_quick`

Use:

```bash
./quick.sh
```

Then summarize the newest result manually or with:

```bash
python3 scripts/experiments/analyze_quick_run.py --results <result-json>
```
