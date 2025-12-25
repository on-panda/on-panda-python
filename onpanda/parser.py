import re
import mxlm
from copy import deepcopy
import mximport
import os
import json

with mximport.inpkg():
    from .token_level_supervision_utils import (
        compute_token_level_supervision,
        unicode_tokenizer,
        apply_ignore_unicode_loss_mask_to_content,
    )
    from .correcting_model.correcting_sft_utils import (
        NextTokenPredictionAsCorrectingBuilder,
    )

HASH_TEMPLATE_PREFIX = "<|hash|>"
HASH_TEMPLATE_REGEX = r"^<\|hash\|>([A-Za-z0-9+\/=]+)$"


def recover_hash_map(data):
    # work on any dialogs
    def recover(obj):
        if isinstance(obj, (dict, list)):
            items = obj.items() if isinstance(obj, dict) else enumerate(obj)
            for key, value in items:
                if isinstance(value, str) and re.match(HASH_TEMPLATE_REGEX, value):
                    hash_value = value.replace(HASH_TEMPLATE_PREFIX, "")
                    obj[key] = data["hash_map"][hash_value]
                elif isinstance(value, (list, dict)):
                    recover(value)

    recover(data["dialogs"])
    data["hash_map"] = {}
    return data


def sequence_prefix_length(seq1, seq2):
    for i in range(min(len(seq1), len(seq2))):
        if seq1[i] != seq2[i]:
            return i
    return i + 1


RESPONSE_ROLES = ["assistant"]


class PandaTreeParser:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def parser(self, data):
        return PandaTree(data, tokenizer=self.tokenizer)


class PandaTree:
    SUPPORT_PANDA_TREE_VERSION = "2.0"

    def __init__(self, data, tokenizer=None):
        self.raw_data = data
        self.tokenizer = tokenizer
        self.data = data = self.pre_process(data)
        assert len(data["dialogs"]), "Empty dialogs: " + str(data)
        dialogs = data["dialogs"]
        dialog_valide_keys = sorted(dialogs)
        dense_keys = [
            k for k in dialog_valide_keys if dialogs[k]["annotate"].get("is_good")
        ]
        assert dense_keys, "No any is_good dialog in this data."

        trees = {}
        to_parent = {}

        def get_parents(dialog_key):  # include self
            parents = [dialog_key]
            while to_parent[parents[-1]]:
                parents.append(to_parent[parents[-1]])
            return parents[::-1]

        for dialog_key in dialog_valide_keys:
            dialog = data["dialogs"][dialog_key]
            operations = dialog.get("operations", [])
            is_tree_root = self.is_operation_tree_root(operations)
            if not is_tree_root:
                parent = int(operations[0]["parent"])
                if parent not in to_parent:
                    # belong to deleted
                    is_tree_root = True
            if is_tree_root:
                trees[dialog_key] = {}
                to_parent[dialog_key] = None
            else:
                to_parent[dialog_key] = parent
                parents = get_parents(parent)

                node = trees
                for _parent in parents:
                    node = node[_parent]
                node[dialog_key] = {}

        # best practice: only do negative supervision for outcome and fork pairs. because dense_keys will provide positive supervision. if do so, negative supervision should duplicate.
        # pair of (negative, positive)
        outcome_pairs = (
            []
        )  # pairs not include token level supervision. similar to DPO pair
        fork_pairs = []

        def flatten_tree(tre):
            if not tre:
                return []
            res = []
            for k in tre:
                res.append(k)
                res.extend(flatten_tree(tre[k]))
            return res

        for tree_key in trees:
            tre = trees[tree_key]  # avoid boxx.tree variable
            flattens = [tree_key] + flatten_tree(tre)

            tree_dense_keys = [
                dialog_key for dialog_key in flattens if dialog_key in dense_keys
            ]

            if tree_dense_keys:  # this tree has dense, only need to do fork pairs
                for dialog_key in flattens:
                    if dialog_key not in dense_keys:
                        # find in this tree's dense which has nearst sequence as pair dense
                        # may be multiple pair dense with the same prefix length
                        min_prefix_len = 9e10
                        pair_dense_keys = []
                        for dense_key in tree_dense_keys:
                            prefix_len = sequence_prefix_length(
                                dialogs[dialog_key]["sequence"],
                                dialogs[dense_key]["sequence"],
                            )
                            # for multiple pieces of data with the same branching point, token level negative supervision should duplicate
                            if prefix_len < min_prefix_len:
                                min_prefix_len = prefix_len
                                pair_dense_keys = [dense_key]
                            elif prefix_len == min_prefix_len:
                                pair_dense_keys.append(dense_key)
                        for dense_key in pair_dense_keys:
                            assert (
                                dialogs[dialog_key]["prompt_hash"]
                                == dialogs[dense_key]["prompt_hash"]
                            ), f"Prompt hash of {pair_dense_keys}, {dense_key} not equal!\n{dialogs[dialog_key]['messages']}\n{dialogs[dense_key]['messages']}"
                        fork_pairs.extend(
                            [(dialog_key, dense_key) for dense_key in pair_dense_keys]
                        )
            else:  # this tree has no dense, need to do outcome pairs
                for dialog_key in flattens:
                    dialog = dialogs[dialog_key]
                    if (dialog.get("operations") or [{}])[0].get("is_prompt_modified"):
                        # when prompt modified and no dense in this tree, don't need to as outcome negative supervision
                        break
                    for dense_key in dense_keys:
                        if dialog["prompt_hash"] == dialogs[dense_key]["prompt_hash"]:
                            outcome_pairs.append((dialog_key, dense_key))

        self.trees = trees
        self.to_parent = to_parent
        self.dense_keys = dense_keys
        self.outcome_pairs = outcome_pairs
        self.fork_pairs = fork_pairs
        self.valid_dialog_keys = dialog_valide_keys
        # g()

    def pre_process(self, data):
        assert "dialogs" in data, "invalid data format."
        data = deepcopy(data)
        data = recover_hash_map(data)
        assert self.SUPPORT_PANDA_TREE_VERSION >= data.get(
            "version", "0.0"
        ), f"Current parser support data version: {self.SUPPORT_PANDA_TREE_VERSION}, panda tree data version: {data['version']} Need to update onpanda package."
        assert (
            "update_time" in data
        ), "Never saved data. Which mean may never checked by Annotator."
        assert len(data["dialogs"]) >= 1, "Empty dialogs!"
        data["dialogs"] = {int(k): v for k, v in data["dialogs"].items()}
        # set default is_good
        max_key = max(data["dialogs"])
        for dialog_key in data["dialogs"]:
            dialog = data["dialogs"][dialog_key]
            if "annotate" not in dialog:
                dialog["annotate"] = {}
            if dialog["annotate"].get("is_good") is None:
                dialog["annotate"]["is_good"] = dialog_key == max_key

        dialog_valide_keys = [
            key
            for key in sorted(data["dialogs"].keys())
            if data["dialogs"][key]["messages"][-1]["role"] in RESPONSE_ROLES
        ]
        data["dialogs"] = {k: data["dialogs"][k] for k in dialog_valide_keys}
        self.prompt_hash_to_keys = {}
        for dialog_key in dialog_valide_keys:
            # set prompt_hash
            dialog = data["dialogs"][dialog_key]
            assert "annotate" in dialog, "No annotate in dialog!"
            prompt = mxlm.remove_last_assistant(dialog["messages"])
            dialog["prompt_hash"] = mxlm.hash_object_sha256_base64(prompt)
            self.prompt_hash_to_keys[dialog["prompt_hash"]] = (
                self.prompt_hash_to_keys.get(dialog["prompt_hash"], []) + [dialog_key]
            )
            dialog["sequence"] = self.messages_to_sequence(dialog["messages"])

            # set operations' parent to int
            for operation in dialog.get("operations", []):
                if "parent" in operation:
                    operation["parent"] = int(operation["parent"])
        return data

    def is_operation_tree_root(self, operations):
        if not operations:
            return True
        operation = operations[0]
        if operation.get("is_new_generated"):
            return True
        if operation.get("is_prompt_modified"):
            return True
        if not operation.get("parent"):
            return True

    def messages_to_sequence(self, messages):
        if not self.tokenizer:
            # default
            return mxlm.messages_to_sequence(messages)
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def __str__(self):
        tree_str = str(self.trees).replace(": {}", ":''")
        s = f"""PandaTree({tree_str}):
    dense_keys: {self.dense_keys}
    fork_pairs: {self.fork_pairs}
    outcome_pairs: {self.outcome_pairs}"""
        return s

    __repr__ = __str__

    def build_legacy_data_v1(self, only_finish_reason_is_stop=False):
        data = self.data
        dialogs = data["dialogs"]
        sfts = []
        for dense_key in self.dense_keys:
            dialog = dialogs[dense_key]
            messages = deepcopy(dialog["messages"])
            # add onpanda
            onpanda_info = {"dialog_key": dense_key}
            if data.get("uuid"):
                onpanda_info["uuid"] = data["uuid"]
            messages[0]["onpanda"] = onpanda_info
            sfts.append(messages)
        preferences = []
        for rejected_key, chosen_key in self.outcome_pairs + self.fork_pairs:
            rejected = dialogs[rejected_key]
            chosen = dialogs[chosen_key]
            assert rejected["prompt_hash"] == chosen["prompt_hash"]
            preference = deepcopy(
                rejected["messages"][:-1]
                + [rejected["messages"][-1], chosen["messages"][-1]]
            )
            # key name 'chosen, rejected' from Anthropic/hh-rlhf
            preference[-1]["preference_tag"] = "chosen"
            preference[-2]["preference_tag"] = "rejected"
            onpanda_info = {"dialog_pair": (rejected_key, chosen_key)}
            onpanda_info["pair_type"] = (
                "fork" if (rejected_key, chosen_key) in self.fork_pairs else "outcome"
            )
            if data.get("uuid"):
                onpanda_info["uuid"] = data["uuid"]
            preference[0]["onpanda"] = onpanda_info
            if only_finish_reason_is_stop:
                if (
                    preference[-1].get("finish_reason") != "stop"
                    or preference[-2].get("finish_reason") != "stop"
                ):
                    continue
            preferences.append(preference)
        return dict(sfts=sfts, preferences=preferences)

    def build_token_level_supervision_data_v1(self, tokenizer=None):
        """
        token_level supervision data v1 structrue like this:
        tree-token_levels[0]
        └── /: list  2
            ├── 0: dict  2
            │   ├── role: user
            │   └── content: 1+1=? using English word
            └── 1: dict  4
                ├── role: assistant
                ├── content: list  3
                │   ├── 0: dict  3
                │   │   ├── type: text
                │   │   ├── text: 1+1=
                │   │   └── ignore_loss: True
                │   ├── 1: dict  4
                │   │   ├── type: text
                │   │   ├── text: two
                │   │   ├── ignore_loss: False
                │   │   └── tokens: list  1
                │   │       └── 0: 99473
                │   └── 2: dict  3
                │       ├── type: text
                │       ├── text: .
                │       └── ignore_loss: True
                ├── finish_reason: stop
                └── token_level: dict  4
                    ├── fork_token_idx: 15
                    ├── chosen_token_id: 99473
                    ├── rejected_token_id: 26288
                    ├── chosen_dialog_key: 3
                    ├── rejected_dialog_key: 2
                    ├── chosen_text: two
                    └── rejected_content: list  3
                        ├── 0: dict  3
                        │   ├── type: text
                        │   ├── text: 1+1=
                        │   └── ignore_loss: True
                        ├── 1: dict  5
                        │   ├── type: text
                        │   ├── text: three
                        │   ├── ignore_loss: False
                        │   ├── tokens: list  1
                        │   │   └── 0: 26288
                        │   └── rejected_loss: True
                        └── 2: dict  3
                            ├── type: text
                            ├── text: .
                            └── ignore_loss: True
        """
        tokenizer = tokenizer or self.tokenizer
        assert (
            tokenizer
        ), "token_level_supervision needs to set tokenizer, or you could using onpanda.unicode_tokenizer"
        token_levels = []
        for rejected_key, chosen_key in self.fork_pairs:
            chosen_msgs = self.data["dialogs"][chosen_key]["messages"]
            rejected_msgs = self.data["dialogs"][rejected_key]["messages"]
            chosen_content = chosen_msgs[-1]["content"]
            rejected_content = rejected_msgs[-1]["content"]
            token_level_info = compute_token_level_supervision(
                chosen_content=chosen_content,
                rejected_content=rejected_content,
                tokenizer=tokenizer,
            )
            token_level_info["version"] = "1.0"
            token_level_info["chosen_dialog_key"] = chosen_key
            token_level_info["rejected_dialog_key"] = rejected_key
            token_level_info["rejected_finish_reason"] = rejected_msgs[-1].get(
                "finish_reason", ""
            )
            token_level_msgs = chosen_msgs[:-1] + [
                {
                    **chosen_msgs[-1],
                    "content": token_level_info.pop("chosen_content"),
                    "token_level": token_level_info,
                }
            ]
            token_level_msgs = deepcopy(token_level_msgs)

            onpanda_info = {"dialog_pair": (rejected_key, chosen_key)}
            if self.data.get("uuid"):
                onpanda_info["uuid"] = self.data["uuid"]
            token_level_msgs[0]["onpanda"] = onpanda_info

            token_levels.append(token_level_msgs)
        # g()
        return token_levels

    def build_token_level_supervision_data_v2(self, tokenizer=None):
        """
        Merge multi token_level_supervision_data_v1 with same chosen_dialog_key into one messages with multi ignore_loss=False tokens (chosen tokens), thus to reduce SFT data number.
        Also gather token_level to token_levels

        Should save as 'xxx.token.json'
        """
        token_level_v1s = self.build_token_level_supervision_data_v1(
            tokenizer=tokenizer
        )
        chosen_dialog_key_to_token_level_v1s = {}
        for token_level_v1 in token_level_v1s:
            chosen_dialog_key = token_level_v1[-1]["token_level"]["chosen_dialog_key"]
            chosen_dialog_key_to_token_level_v1s.setdefault(
                chosen_dialog_key, []
            ).append(token_level_v1)
        token_level_v2s = []
        for chosen_dialog_key in sorted(chosen_dialog_key_to_token_level_v1s):
            msg = None
            ignore_loss_unicode_masks = []
            token_level_infos = []
            token_level_v1s = chosen_dialog_key_to_token_level_v1s[chosen_dialog_key]
            for token_level_v1 in sorted(
                token_level_v1s,
                key=lambda msgs: msgs[-1]["token_level"]["fork_token_idx"],
            ):
                if msg is None:
                    msg = deepcopy(token_level_v1[-1])
                    content_str0 = "".join([c["text"] for c in msg.pop("content")])
                    msg["token_levels"] = [msg.pop("token_level")]

                content = token_level_v1[-1]["content"]
                content_str = "".join([c["text"] for c in content])
                assert (
                    content_str == content_str0
                ), f"Chosen content string not equal: {content_str} != {content_str0}"
                ignore_loss_unicode_mask = sum(
                    [[c.get("ignore_loss", False)] * len(c["text"]) for c in content],
                    [],
                )
                ignore_loss_unicode_mask += [
                    content[-1].get("ignore_loss", False)
                ]  # concat stop token's mask
                ignore_loss_unicode_masks.append(ignore_loss_unicode_mask)
                token_level_infos.append(token_level_v1[-1]["token_level"])
            # ignore_loss_unicode_masks = np.array(ignore_loss_unicode_masks)
            # ignore_loss_unicode_mask_merged = ~(~ignore_loss_unicode_masks).any(0)
            ignore_loss_unicode_mask_merged = [
                all(row[i] for row in ignore_loss_unicode_masks)
                for i in range(len(ignore_loss_unicode_masks[0]))
            ]
            chosen_content_merged = apply_ignore_unicode_loss_mask_to_content(
                ignore_loss_unicode_mask_merged, content_str0
            )
            token_level_v2 = deepcopy(token_level_v1)
            token_level_v2[-1]["content"] = chosen_content_merged
            token_level_v2[-1]["token_levels"] = token_level_infos
            token_level_v2[-1].pop("token_level")
            token_level_v2s.append(token_level_v2)
        # g() / 0
        return token_level_v2s

    def build_correcting_sft_data_v1(self, ntp_as_correcting_builder):
        """
        tree(correcting_sfts[-1])
        ├── 1: dict  3
        │   ├── role: user
        │   ├── content: How many 1 in 01011010101111011011?
        │   └── description: Answer is 13
        ├── 2: dict  5
        │   ├── role: assistant
        │   ├── ignore_loss: True
        │   ├── content: To determine how many times the digit 1 ap...
        │   ├── finish_reason: stop
        │   └── token_level: dict  11
        ├── 3: dict  2
        │   ├── role: system
        │   └── content: {correcting SFT system prompt}...
        └── 4: dict  3
            ├── role: assistant
            ├── content: <|fim_pad|> **0<|fim_pad|>0<|fim_pad|> <|f...
            └── correcting: dict  7
                ├── location_text:  **0
                ├── match_num: 1
                ├── location_index: 0
                ├── location_tokens: list  2
                │   ├── 0: 3070
                │   └── 1: 15
                ├── replacement_text:
                ├── is_good: False
                └── scope_slice: tuple 2
                    ├── 0: -1
                    └── 1: None
        """
        sfts = self.build_legacy_data_v1()["sfts"]
        correcting_sfts = [
            ntp_as_correcting_builder.build_correcting_sft_by_token_level_SFT(
                sft, is_good=True
            )
            for sft in sfts
        ]
        token_level_v1s = self.build_token_level_supervision_data_v1(
            tokenizer=ntp_as_correcting_builder.tokenizer
        )
        correcting_sfts += [
            ntp_as_correcting_builder.build_correcting_sft_by_token_level_SFT(
                sft, is_good=False
            )
            for sft in token_level_v1s
        ]
        return correcting_sfts


def build_test_panda_tree(panda_json=None, tokenizer=None):
    if panda_json is None:
        panda_json = "../../on-panda-example-data/panda_json/2025-08-19_how-many-1s_tokenizer-Qwen2.5.panda.json"
    if isinstance(panda_json, str):
        panda_json = json.load(open(panda_json))
    panda_tree = PandaTree(panda_json, tokenizer=tokenizer)
    return panda_tree


if __name__ == "__main__":
    from boxx import *  # pip install boxx

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # tokenizer = None
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    )
    # %%

    # tokenizer = unicode_tokenizer
    # test_json = "../../asset/on-panda-example/shape-of-V-test-hash.panda.json"
    # test_json = "../../asset/on-panda-example/how-many-1s.panda.json"
    # test_json = "../../on-panda-example-data/panda_json/2025-04-12_Chinese_acrostic_poem_藏头诗_tokenizer-step2.panda.json"
    # test_json = "../../on-panda-example-data/panda_json/2025-09-10_correcting_sft_tokenizer-Qwen2.5.panda.json"
    test_json = "../../on-panda-example-data/panda_json/2025-08-19_how-many-1s_tokenizer-Qwen2.5.panda.json"

    panda_tree = build_test_panda_tree(test_json, tokenizer)
    print(panda_tree)
    tree(panda_tree.trees)

    legacy_data = panda_tree.build_legacy_data_v1()
    sfts = legacy_data["sfts"]
    # tree(legacy_data)

    token_level_v1s = panda_tree.build_token_level_supervision_data_v1(
        tokenizer=tokenizer
    )

    token_level_v2s = panda_tree.build_token_level_supervision_data_v2(
        tokenizer=tokenizer
    )
    sft_correcting_builder = NextTokenPredictionAsCorrectingBuilder(
        tokenizer=tokenizer,
        SPLIT_TOKEN="<|fim_pad|>",  # for qwen 2.5
        STOP_TOKEN="<|fim_suffix|>",
        max_location_tokens=20,
    )

    correcting_sfts = panda_tree.build_correcting_sft_data_v1(sft_correcting_builder)

    tree(correcting_sfts[-1])
    # savejson(correcting_sfts[-1], "/home/yl/onPanda/asset/correcting_sft/correcting_sft_example1.sft.json")
    # tree(correcting_sfts[3])
