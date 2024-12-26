import mxlm


class PandaTreePaser:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
    
    def messages_to_sequence(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False)
        
    def paser(self, panda_tree):
        assert 'update_time' in panda_tree, "Never saved data. Which mean may never checked by Annotator."
        assert len(panda_tree['dialogs']) >= 1, "Empty dialogs!"
        panda_tree['dialogs'] = {int(k): v for k, v in panda_tree['dialogs'].items()}
        dialog_ids = sorted(panda_tree['dialogs'].keys())
        dialogs = panda_tree['dialogs']
        prompt_hash_to_ids = {}
        for dialog_id in dialog_ids:
            dialog = dialogs[dialog_id]
            assert 'annotate' in dialog, "No annotate in dialog!"
            prompt = mxlm.remove_last_assistant(dialog['messages'])
            dialog['prompt_hash'] = mxlm.hash_object_sha256_base64(prompt)
            prompt_hash_to_ids[dialog['prompt_hash']] = prompt_hash_to_ids.get(dialog['prompt_hash'], []) + [dialog_id]
            dialog['sequence'] = self.messages_to_sequence(dialog['messages'])

        
        dense_ids = [k for k in dialog_ids if dialogs[k]["annotate"].get("is_good")]


        trees = {}
        to_parent = {}
        def get_parents(dialog_id):  # include self
            parents = [dialog_id]
            while to_parent[parents[-1]]:
                parents.append(to_parent[parents[-1]])
            return parents[::-1]
            
        for dialog_id in dialog_ids:
            dialog = panda_tree['dialogs'][dialog_id]
            operations = dialog.get('operations', [])
            is_tree_root = self.is_operation_tree_root(operations)
            if not is_tree_root:
                parent = int(operations[0]['parent'])
                if parent not in to_parent:
                    # belong to deleted
                    is_tree_root = True
            if is_tree_root:
                trees[dialog_id] = {}
                to_parent[dialog_id] = None
            else:
                to_parent[dialog_id] = parent
                parents = get_parents(parent)
                
                node = trees
                for _parent in parents:
                    node = node[_parent]
                node[dialog_id] = {}

        outcome_pairs = []
        fork_pairs = []
        def flatten_tree(tre):
            if not tre:
                return []
            res = []
            for k in tre:
                res.append(k)
                res.extend(flatten_tree(tre[k]))
            return res
        for tree_id in trees:
            tre = trees[tree_id]  # avoid boxx.tree variable
            has_dense = False
            flattens = [tree_id] + flatten_tree(tre)
            for dialog_id in flattens:
                if dialog_id in dense_ids:
                    has_dense = True
                    break
            if has_dense:
                for dialog_id in flattens:
                    if dialog_id not in dense_ids:
                        fork_pairs.append((tree_id, dialog_id))
            else:
                pass

                            
        g()
        
    def is_operation_tree_root(self, operations):
        if not operations:
            return True
        operation = operations[0]
        if operation.get('is_new_generated'):
            return True
        if operation.get('is_prompt_modified'):
            return True
        if not operation.get('parent'):
            return True
        


if __name__ == "__main__":
    from boxx import *
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer

    # tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    # tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-bnb-4bit")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    parser = PandaTreePaser(tokenizer)

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

    chatml = tokenizer.apply_chat_template(messages, tokenize=False)
    print(chatml)
    data = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        return_assistant_tokens_mask=True,
    ).data

    import json
    pt = panda_tree = json.load(open('../../asset/on-panda-example/how-many-1s.panda.json'))
    
    parser.paser(panda_tree)
    print(panda_tree)
