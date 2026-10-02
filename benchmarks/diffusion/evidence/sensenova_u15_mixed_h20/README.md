# SenseNova-U1.5 mixed warmup: H20 run evidence

These are the original benchmark JSON reports from the H20-3e runs described in
[the SenseNova recipe](../../../../recipes/SenseNova/SenseNova-U1.5.md).
The benchmark sent eight serial requests per run: two repetitions of
`t2i:1024x1024`, `t2t`, `i2t`, `t2i:1536x1536`.
The original reports predate the `image_seed` report field added during review;
the benchmark payload used the fixed image seed 42 in all four runs.
The warmup reports say `server_sha=a038b3817+working-tree` because the
implementation was still uncommitted during measurement; it was subsequently
committed as `37ac759` in this PR.

| Run | Original report | Start (Unix s) | First `/health` (Unix s) | Startup |
| --- | --- | ---: | ---: | ---: |
| base BF16, default warmup | [base2.json](base2.json) | 1790845158.859599352 | 1790845184.847380877 | 26.0 s |
| base BF16, mixed warmup | [warm.json](warm.json) | 1790845274.806712151 | 1790845365.469507933 | 90.7 s |
| distilled LoRA, default warmup | [base-distill.json](base-distill.json) | 1790845784.823287725 | 1790845820.939787865 | 36.1 s |
| distilled LoRA, mixed warmup | [warm-distill.json](warm-distill.json) | 1790845938.858382225 | 1790846034.511888742 | 95.7 s |

The startup durations come from the saved server-start and first successful
`/health` timestamps. The process GPU memory values in the recipe were
observed with `nvidia-smi` at readiness, but a raw `nvidia-smi` snapshot
was not saved; those values are approximate author observations.

## Server log excerpts

These lines come from the corresponding original server logs. They show whether
the paged decode graph was captured before or after readiness. The JSON reports
count captures and recompiles only after the benchmark starts.

### base BF16, default warmup

```text
(DiffusionWorker pid=490342) INFO:     Application startup complete.
(DiffusionWorker pid=490342) DEBUG 10-01 17:00:54 [paged_decode.py:357] Captured decode graph for bucket=512 generation=0
```

### base BF16, mixed warmup

```text
(DiffusionWorker pid=494307) DEBUG 10-01 17:02:42 [paged_decode.py:357] Captured decode graph for bucket=512 generation=0
(DiffusionWorker pid=494307) INFO 10-01 17:02:42 [pipeline_sensenova_u1.py:1379] SenseNova mixed warmup text_to_text took 63.02 s
(DiffusionWorker pid=494307) INFO 10-01 17:02:42 [pipeline_sensenova_u1.py:1379] SenseNova mixed warmup image_to_text took 0.08 s
(DiffusionWorker pid=494307) INFO 10-01 17:02:43 [pipeline_sensenova_u1.py:1379] SenseNova mixed warmup text_to_image_1024x1024 took 0.86 s
(DiffusionWorker pid=494307) INFO 10-01 17:02:44 [pipeline_sensenova_u1.py:1379] SenseNova mixed warmup text_to_image_1536x1536 took 0.96 s
(DiffusionWorker pid=494307) INFO:     Application startup complete.
```

### distilled LoRA, default warmup

```text
(DiffusionWorker pid=511203) INFO:     Application startup complete.
(DiffusionWorker pid=511203) DEBUG 10-01 17:11:31 [paged_decode.py:357] Captured decode graph for bucket=512 generation=0
```

### distilled LoRA, mixed warmup

```text
(DiffusionWorker pid=516962) DEBUG 10-01 17:13:52 [paged_decode.py:357] Captured decode graph for bucket=512 generation=0
(DiffusionWorker pid=516962) INFO 10-01 17:13:52 [pipeline_sensenova_u1.py:1382] SenseNova mixed warmup text_to_text took 58.80 s
(DiffusionWorker pid=516962) INFO 10-01 17:13:52 [pipeline_sensenova_u1.py:1382] SenseNova mixed warmup image_to_text took 0.07 s
(DiffusionWorker pid=516962) INFO 10-01 17:13:52 [pipeline_sensenova_u1.py:1382] SenseNova mixed warmup text_to_image_1024x1024 took 0.67 s
(DiffusionWorker pid=516962) INFO 10-01 17:13:53 [pipeline_sensenova_u1.py:1382] SenseNova mixed warmup text_to_image_1536x1536 took 0.61 s
(DiffusionWorker pid=516962) INFO:     Application startup complete.
```

No `Recompiling function` line appeared during any benchmark sequence.
