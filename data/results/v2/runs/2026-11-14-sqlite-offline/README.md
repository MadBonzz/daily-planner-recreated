# SQLite history replay and offline cautions

This run used the ten checked-in synthetic ActivityWatch databases and the current fallback estimator. The task histories supplied their authored interval labels. No model calls were made. Each database's SHA-256 digest is recorded in the corresponding history file.

Across the 20 holdouts, the fallback's mean absolute error was 13.60 minutes. Fifteen actual durations fell within the forecast range. The 20 point estimates matched the [earlier offline run](../2026-11-14-offline/results.json) case by case. The updated cautions cited a prespecified historical source on four of five applicable cases, up from zero in the earlier run. The four were the crash-recovery proof, scheduling call with Mara, rover run, and Fourier lecture. The migration repair had no caution. This metric measures citation overlap. It does not assess how helpful a particular warning sounded to a user.

[Case forecasts and citations](results.json)

```sh
python -m examples.estimation.run evaluate --offline --json data/results/local/sqlite-offline-01/results.json
```
