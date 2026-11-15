"""Offline correction integration checks for model-owned response templates."""

if __name__ == "__main__":
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    import mximport
    import mxlm
    from mxlm.response_templates import (
        DefaultResponseTemplate,
        DeepSeekV4ResponseTemplate,
        DeepSeekV4p1ResponseTemplate,
    )

    with mximport.inpkg():
        from .far_correction_utils import FindAndReplaceCorrectionAdapter
        from .correcting_model import CorrectingModel
        from ..token_level_supervision_utils import utf8_tokenizer
        from ..test_utils import get_test_trajectories

    for template_class in (DeepSeekV4ResponseTemplate, DeepSeekV4p1ResponseTemplate):
        policy = mxlm.ChatAPI(
            base_url="http://policy/v1", model=template_class.default_name_or_path
        )
        template = policy.response_template
        assert isinstance(template, template_class)
        for trajectory in get_test_trajectories().values():
            for correction in trajectory["corrections"]:
                fork = correction["fork_message_index"]
                if fork is None:
                    continue
                partial = correction["partial_messages"][fork]
                text = template.apply(partial)["templated_prompt"]
                parsed = template.parse(
                    text,
                    messages=trajectory["messages"][:fork],
                    tools=trajectory.get("tools"),
                    finish_reason=partial.get("finish_reason"),
                )
                assert template.apply(parsed)["templated_prompt"] == text

        adapter = FindAndReplaceCorrectionAdapter(
            tokenizer=utf8_tokenizer, max_replacement_tokens=1000
        )
        correcting_model = CorrectingModel(SimpleNamespace(model="correcting"), adapter)
        adapter_policy = FindAndReplaceCorrectionAdapter(
            tokenizer=utf8_tokenizer,
            max_replacement_tokens=1000,
            response_template=template,
        )
        user_message = dict(role="user", content="q")
        rejected = dict(role="assistant", content="wrong", finish_reason="stop")
        tool_partial = dict(
            role="assistant",
            tool_calls=[dict(function=dict(name="read", arguments='{"path": "/tm'))],
        )
        for partial, reasoning_delta, content_delta, expected in (
            (
                dict(role="assistant"),
                "think",
                "answer",
                dict(reasoning="think", content="answer"),
            ),
            (
                dict(role="assistant", reasoning="start"),
                " end",
                "answer",
                dict(reasoning="start end", content="answer"),
            ),
            (
                dict(role="assistant", reasoning="think", content="start"),
                " end",
                "!",
                dict(reasoning="think", content="start end!"),
            ),
            (
                tool_partial,
                "p",
                template.parameter_end
                + "\n"
                + template.invoke_end
                + "\n"
                + template.tool_calls_end,
                dict(content="", finish_reason="tool_calls"),
            ),
        ):
            prefix = adapter_policy.build_partial_templated_prompt(rejected, partial)[
                "templated_prompt"
            ]
            correcting_model.correct = Mock(
                return_value=dict(
                    correction=dict(
                        messages_location=dict(path_keys=[1, "content"], char_index=0)
                    ),
                    partial_messages=[user_message, partial],
                )
            )
            policy.client.chat.completions.create = Mock(
                return_value=SimpleNamespace(
                    dict=lambda: dict(
                        choices=[
                            dict(
                                finish_reason="stop",
                                message=dict(
                                    role="assistant",
                                    content=content_delta,
                                    reasoning_content=reasoning_delta,
                                ),
                            )
                        ]
                    )
                )
            )
            # Correcting an earlier turn must discard the old conversation suffix.
            corrected_messages = correcting_model.correct_and_rollout(
                [
                    user_message,
                    rejected,
                    dict(role="user", content="old follow-up"),
                    dict(role="assistant", content="old answer", finish_reason="stop"),
                ],
                policy,
                adapter_policy=adapter_policy,
            )["corrected_messages"]
            assert len(corrected_messages) == 2
            assert corrected_messages[0] == user_message
            corrected = corrected_messages[-1]
            for key, value in expected.items():
                assert corrected[key] == value, corrected
            if partial is tool_partial:
                assert corrected["tool_calls"][0]["function"] == dict(
                    name="read", arguments='{"path": "/tmp"}'
                )
            request = policy.client.chat.completions.create.call_args.kwargs
            assert request["messages"] == [
                user_message,
                dict(role="assistant", content=prefix, prefix=True),
            ]
            assert request["continue_final_message"] is True
            assert request["add_generation_prompt"] is False
            assert request["skip_special_tokens"] is False
        print(template_class.__name__, "passed: correction trajectories, continuation")

    # Default handles plain content even when ChatAPI has no native template.
    # An open structured channel must not acquire an empty downstream content channel.
    for name_or_path, partial, delta in (
        (
            None,
            dict(role="assistant", content="hi!"),
            dict(content="!"),
        ),
        (
            "onpanda/correction-call",
            dict(role="assistant", reasoning="start"),
            dict(reasoning_content=" end", content=None),
        ),
    ):
        policy = mxlm.ChatAPI(model="deployment", name_or_path=name_or_path)
        # Without adapter_policy, reuse the correcting adapter as configured.
        correcting_model = CorrectingModel(
            SimpleNamespace(model="correcting"),
            FindAndReplaceCorrectionAdapter(
                max_replacement_tokens=1000, response_template=policy.response_template
            ),
        )
        correcting_model.correct = Mock(
            return_value=dict(
                correction=dict(
                    messages_location=dict(path_keys=[1, "content"], char_index=0)
                ),
                partial_messages=[user_message, partial],
            )
        )
        policy.client.chat.completions.create = Mock(
            return_value=SimpleNamespace(
                dict=lambda: dict(
                    choices=[
                        dict(
                            message=dict(role="assistant", **delta),
                            finish_reason="length",
                        )
                    ]
                )
            )
        )
        corrected = correcting_model.correct_and_rollout(
            [user_message, rejected], policy
        )["corrected_messages"][-1]
        if name_or_path is None:
            assert policy.response_template is None
            assert corrected["content"] == partial["content"] + "!"
            assert "reasoning" not in corrected
        else:
            assert corrected["reasoning"] == "start end"
            assert "content" not in corrected
        assert corrected["finish_reason"] == "length"
    print("Plain content and open correction-call reasoning passed")

    # Do not inject the separator before a model can emit a merged newline token.
    trajectory = get_test_trajectories("bad_argument_key")
    adapter = FindAndReplaceCorrectionAdapter(
        tokenizer=utf8_tokenizer, max_replacement_tokens=5
    )
    for model in ("Qwen/Qwen3.5-35B-A3B", "stepfun-ai/Step-3.7-Flash"):
        policy = mxlm.ChatAPI(base_url="http://policy/v1", model=model)
        correcting_model = CorrectingModel(SimpleNamespace(model="correcting"), adapter)
        correcting_model.correct = Mock(
            return_value=adapter.apply(
                trajectory["messages"],
                trajectory["corrections"][0]["default_far_ref"],
                tools=trajectory["tools"],
            )
        )
        adapter_policy = FindAndReplaceCorrectionAdapter(
            tokenizer=utf8_tokenizer,
            max_replacement_tokens=5,
            response_template=policy.response_template,
        )
        prefix = adapter_policy.build_partial_templated_prompt(
            trajectory["messages"][-1],
            correcting_model.correct.return_value["partial_messages"][-1],
        )["templated_prompt"]
        assert prefix.endswith("<parameter=path>")
        policy.client.chat.completions.create = Mock(
            return_value=SimpleNamespace(
                dict=lambda: dict(
                    choices=[
                        dict(
                            finish_reason="stop",
                            message=dict(
                                role="assistant",
                                content="\n\ntext\n</parameter>\n</function>\n</tool_call>",
                            ),
                        )
                    ]
                )
            )
        )
        corrected = correcting_model.correct_and_rollout(
            trajectory["messages"],
            policy,
            adapter_policy=adapter_policy,
            tools=trajectory["tools"],
        )["corrected_messages"][-1]
        request = policy.client.chat.completions.create.call_args.kwargs
        assert request["messages"][-1]["content"] == prefix
        assert (
            corrected["tool_calls"][0]["function"]["arguments"] == '{"path": "\\ntext"}'
        )
        assert corrected["finish_reason"] == "tool_calls"
    print("Parameter tag prefixes and leading value newlines passed")
