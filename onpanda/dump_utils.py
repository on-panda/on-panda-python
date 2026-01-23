import mxlm
import json
from copy import deepcopy
from mximport import inpkg

with inpkg():
    from .utils import HASH_MAP_LENGTH, HASH_TEMPLATE_PREFIX


def using_hash_map_remove_redundancy(dumped):
    # if message's content length or chunk's content length > HASH_MAP_LENGTH,
    # use hash map to remove redundancy
    # same as on-panda-vue/src/stores/pandaState.js
    dumped.setdefault("hash_map", {})
    dialogs = dumped.get("dialogs", {})
    for dialog_key in dialogs:
        dialog = dialogs[dialog_key]
        for message in dialog.get("messages", []):
            content = message.get("content")
            if isinstance(content, str) and len(content) > HASH_MAP_LENGTH:
                hash_value = mxlm.hash_object_sha256_base64(content)
                dumped["hash_map"][hash_value] = content
                message["content"] = HASH_TEMPLATE_PREFIX + hash_value
            elif isinstance(content, list):
                for chunk in content:
                    if not isinstance(chunk, dict) or "type" not in chunk:
                        continue
                    chunk_content = chunk.get(chunk["type"])
                    if (
                        chunk_content is not None
                        and len(json.dumps(chunk_content, ensure_ascii=False))
                        > HASH_MAP_LENGTH
                    ):
                        hash_value = mxlm.hash_object_sha256_base64(chunk_content)
                        dumped["hash_map"][hash_value] = chunk_content
                        chunk[chunk["type"]] = HASH_TEMPLATE_PREFIX + hash_value
                    if "blob_url" in chunk:
                        del chunk["blob_url"]
    return dumped


def dump_panda_json(self, path=None):
    dumped = deepcopy(self.data)
    dialogs = dumped.get("dialogs", {})
    dumped_dialogs = {}
    for dialog_key, dialog in dialogs.items():
        dialog = deepcopy(dialog)
        dialog.pop("prompt_hash", None)
        dialog.pop("sequence", None)
        for operation in dialog.get("operations", []):
            if "parent" in operation and operation["parent"] is not None:
                operation["parent"] = str(operation["parent"])
        dumped_dialogs[str(dialog_key)] = dialog
    dumped["dialogs"] = dumped_dialogs

    if isinstance(dumped.get("deleted_dialogs"), dict):
        dumped["deleted_dialogs"] = {
            str(k): v for k, v in dumped["deleted_dialogs"].items()
        }

    dumped = using_hash_map_remove_redundancy(dumped)

    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dumped, f, indent=2, ensure_ascii=False)
        return path
    return dumped


def _json_size_bytes(obj):
    return "%.3f KB" % (len(json.dumps(obj, ensure_ascii=False).encode("utf-8")) / 1024)


def _strip_prompt_and_sequence(data):
    data = deepcopy(data)
    for dialog in data.get("dialogs", {}).values():
        dialog.pop("prompt_hash", None)
        dialog.pop("sequence", None)
    return data


def test_dump_size_compare():
    from onpanda.parser import PandaTree

    path = "../../on-panda-example-data/panda_json/2025-08-19_how-many-1s_tokenizer-Qwen2.5.panda.json"
    path = "../../on-panda-example-data/panda_json/2025-04-12_Chinese_acrostic_poem_藏头诗_tokenizer-step2.panda.json"
    # need multi dialogs with long contex or base64
    # path = "/home/yl/Downloads/d04-i0284-能学到查找token重复增加token，有RLHF概率-跨年热闹氛围与庆祝方式.panda.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    tree = PandaTree(raw)
    dumped = tree.dump()

    print("raw:", _json_size_bytes(_strip_prompt_and_sequence(raw)))
    print("data_json_bytes:", _json_size_bytes(_strip_prompt_and_sequence(tree.data)))
    print("dump_bytes:", _json_size_bytes(dumped))
    import boxx.g


if __name__ == "__main__":
    test_dump_size_compare()
