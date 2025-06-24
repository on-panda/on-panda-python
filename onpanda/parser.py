import re
import mxlm
from copy import deepcopy


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
            self.prompt_hash_to_keys[
                dialog["prompt_hash"]
            ] = self.prompt_hash_to_keys.get(dialog["prompt_hash"], []) + [dialog_key]
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


if __name__ == "__main__":
    from boxx import *  # pip install boxx
    import os
    import json

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = None
    from transformers import AutoTokenizer

    # tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    # tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-bnb-4bit")
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot",
        },
        {
            "role": "user",
            "content": "How many helicopters can a human eat in one sitting?",
        },
        {"role": "assistant", "content": "2 helicopters"},
    ]
    if tokenizer:
        chatml = tokenizer.apply_chat_template(messages, tokenize=False)
        print(chatml)
        data = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            return_assistant_tokens_mask=True,
        ).data

    test_json = "../../asset/on-panda-example/how-many-1s.panda.json"
    # test_json = "../../asset/on-panda-example/shape-of-V-test-hash.panda.json"
    test_json = "../../asset/on-panda-example/parse_example.panda.json"

    panda_json = json.load(open(test_json))

    panda_tree = PandaTree(panda_json, tokenizer=tokenizer)
    legacy_data = panda_tree.build_legacy_data_v1()
    sfts = legacy_data["sfts"]
    print(panda_tree)
    tree - legacy_data
