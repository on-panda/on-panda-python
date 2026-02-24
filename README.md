# `onpanda`: The Companion Python Package for onPanda

### Contents: [Features](#-features) | [Install](#-install) | [Example Data](#-example-data) | [Quick Start](#-quick-start) | [Main Modules](#-main-modules) | [Iterative Correction API](#-iterative-correction-api) | [Data Assumptions](#-data-assumptions)

## ▮ Features
- [x] Parse `.panda.json` into SFT and preference-pair data (`build_legacy_data_v1`)
- [x] Build token-level supervision data (`build_token_level_supervision_data_v1/v2`)
- [x] Build Find-and-Replace correction training data (`build_far_correction_data_v1`)
- [x] Verify and score Find-and-Replace outputs (`FindAndReplaceVerifier`)
- [x] Run iterative correction as a Proxy API (`onpanda.server.iterative_correction_api`)
- [x] Build `panda battle` data from two arena result sets (`build_panda_battle`)

## ▮ Install

```bash
pip install onpanda -U

# Or want to run demos.
git clone https://github.com/on-panda/onpanda.git
cd onpanda
pip install -e .
```

If you want to use tokenizers, install `transformers` separately.

**Example Data**

`on-panda-example-data` is the example dataset repo for this project:

```bash
git clone https://github.com/on-panda/on-panda-example-data.git ../on-panda-example-data
ls ../on-panda-example-data/panda_json/
```

## ▮ Quick Start

```python
import onpanda

panda_path = (
    "../on-panda-example-data/panda_json/"
    "2025-08-19_how-many-1s_tokenizer-Qwen2.5.panda.json"
)
tokenizer=onpanda.unicode_tokenizer
# Use built-in unicode_tokenizer for a minimal runnable flow.
tree = onpanda.PandaTree(panda_path, tokenizer)

# 1) SFT + preference pairs
legacy = tree.build_legacy_data_v1()
print("sfts:", len(legacy["sfts"]))
print("preferences:", len(legacy["preferences"]))

# 2) Token-level supervision
token_level_v1 = tree.build_token_level_supervision_data_v1(
    tokenizer
)
print("token_level_v1:", len(token_level_v1))

# 3) Find-and-Replace correction data
adapter = onpanda.FindAndReplaceCorrectionAdapter(
    tokenizer
)
correction_data = tree.build_far_correction_data_v1(adapter)
print("correction_data:", len(correction_data))
```

Build from plain chat messages:

```python
import onpanda

messages = [
    {"role": "user", "content": "5+7=?"},
    {"role": "assistant", "content": "12"},
]
panda_json = onpanda.messages_to_panda_tree(messages, uuid="demo")
# dump to xxx.panda.json
```

## ▮ Main Modules

- `onpanda/parser.py`: `PandaTree` and data conversion entrypoints
- `onpanda/token_level_supervision_utils.py`: token-level patch extraction and masks
- `onpanda/correcting_model/far_correction_utils.py`: FAR data builder and apply logic
- `onpanda/correcting_model/verifier.py`: FAR parser/locator/reward computation
- `onpanda/correcting_model/correcting_model.py`: iterative correction workflow
- `onpanda/server/iterative_correction_api.py`: Flask wrapper for correction service
- `onpanda/arena/panda_battle.py`: build battle-style comparison data

## ▮ Iterative Correction API
Launch a proxy API server that return response using iterative_correction
```bash
python -m onpanda.server.iterative_correction_api --help
```

## ▮ Data Assumptions
- `PandaTree` is a parser for qualified, annotated Panda JSON.
- `PandaTree` preprocessing currently assumes:
    - Top-level field `dialogs` exists
    - Top-level field `update_time` exists
    - At least one dialog ends with an `assistant` message
    - If `annotate.is_good` is missing, latest dialog is treated as default good

