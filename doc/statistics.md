# onPanda Production Statistics

The tool computes VLM, audio, or agentic production statistics and writes both
aggregate and per-panda-json results. Run commands from the repository root.

Install the optional tokenizer dependency before the first run:

```bash
pip install transformers
```

VLM:

```bash
onpanda/tool/run_statistics.sh doc/vlm_statistics_config.json
```

Audio:

```bash
onpanda/tool/run_statistics.sh doc/audio_statistics_config.json
```

Agentic:

```bash
onpanda/tool/run_statistics.sh doc/agentic_statistics_config.json
```

Pass a replica count for large runs. The wrapper validates every shard before
writing the final aggregate files:

```bash
onpanda/tool/run_statistics.sh doc/audio_statistics_config.json 32
```

Each configured output directory contains:

- `summary.md`: human-readable table with the requested metric names.
- `summary.csv`: `metric_name,value,note`, for copying into paper tables.
- `summary.json`: full structured aggregate statistics, including coverage and filters.
- `samples.jsonl`: one lightweight metrics record per panda json.

Tokenizer:

- Default formal tokenizer is Qwen2.5.
- The loader first tries `Qwen/Qwen2.5-7B-Instruct` from the local cache.
- If it is not cached, the loader tries another locally cached Qwen2.5 tokenizer,
  such as `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`.
- Tokenizer loading is local-cache only. Missing tokenizer files fail loudly.

Important rules:

- `annotate.is_good is None` follows `PandaTree`: the latest dialog is treated
  as `is_good=True`.
- Annotation elapsed time uses `update_time - earliest generation operation`
  across active and deleted dialogs. No idle-gap deduction is applied; sessions
  over 7 hours are excluded from time aggregates.
- Person-month equivalents use 160 elapsed hours per month.
- SFT samples, token-level corrections, and click/edit/regeneration operations are
  divided by all panda-json sessions. A session with no `is_good` contributes zero.
- Tool calls and user turns first take the maximum cumulative count among a
  session's `is_good` responses, then average the session-level maxima.
- Annotator counts use distinct non-empty `sado_info.label_user` values.
- Active dialogs define positive/negative samples and final outputs. Deleted
  dialogs provide referenced ancestors and recorded operation history.
- A new generation or prompt change starts a new response lineage. A correction
  with a missing parent is excluded from final-path and token attribution.
- Manual edits exceeding 80% of the final response tokens are excluded from the
  annotator-input token metric.
- VLM and audio responses only have `content`; `reasoning` and `tool_call`
  correction counts are reported as `0`.
- Agentic token order is `reasoning`, `</think>`, `content`, then tool calls.
- `run_tool_calls` and `start_new_round` start a new response lineage and are not
  counted as regeneration or correction operations.
- A dialog with multiple correction operations cannot be reconstructed from the
  retained snapshots, so its path and token attribution are unavailable.
- Multimodal resources, including tool responses, are deduplicated by content
  hash within each session. Unsupported modalities are reported separately.

Run the focused tests with:

```bash
python3 -m unittest discover -s tests -p 'test_statistics.py'
```
