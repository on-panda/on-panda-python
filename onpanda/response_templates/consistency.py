import json
from copy import deepcopy

import mximport

with mximport.inpkg():
    from ..token_level_supervision_utils import build_tokenizer


template_test_trajectory = dict(
    messages=[
        dict(role="system", content="_system_content1_"),
        dict(role="user", content="_user_content1_"),
        dict(
            role="assistant",
            reasoning="_assistant_reasoning1_",
            content="_assistant_content1_",
            finish_reason="stop",
        ),
        dict(role="user", content="_user_content2_"),
        dict(
            role="assistant",
            reasoning="_assistant_reasoning2_",
            content="_assistant_content2_",
            tool_calls=[
                dict(
                    id="functions._function_name1_:0",
                    type="function",
                    index=0,
                    function=dict(
                        name="_function_name1_",
                        arguments=json.dumps(
                            {
                                "_argument_name1_": "_argument_value1_",
                                "_argument_name2_": "_argument_enum1_",
                            }
                        ),
                    ),
                ),
                dict(
                    id="functions._function_name2_:1",
                    type="function",
                    index=1,
                    function=dict(
                        name="_function_name2_",
                        arguments=json.dumps(
                            {
                                "_argument_name3_": ["_argument_value2_"],
                                "_argument_name4_": {
                                    "_argument_name5_": "_argument_value3_"
                                },
                                "_argument_name6_": "_argument_enum3_",
                            }
                        ),
                    ),
                ),
                dict(
                    id="functions._function_name3_:2",
                    type="function",
                    index=2,
                    function=dict(
                        name="_function_name3_",
                        arguments=json.dumps(
                            {
                                "_argument_number_": 2609.24983,
                                "_argument_integer_": 2609,
                                "_argument_boolean_": True,
                                "_argument_null_": None,
                            }
                        ),
                    ),
                ),
            ],
            finish_reason="tool_calls",
        ),
        dict(
            role="tool",
            content="_tool_content1_",
            tool_call_id="functions._function_name1_:0",
            name="_function_name1_",
        ),
        dict(
            role="tool",
            content="_tool_content2_",
            tool_call_id="functions._function_name2_:1",
            name="_function_name2_",
        ),
        dict(
            role="tool",
            content="_tool_content3_",
            tool_call_id="functions._function_name3_:2",
            name="_function_name3_",
        ),
        dict(
            role="assistant",
            reasoning="_assistant_reasoning3_",
            content="_assistant_content3_",
            finish_reason="stop",
        ),
    ],
    tools=[
        dict(
            type="function",
            function=dict(
                name="_function_name1_",
                description="_function_description1_",
                parameters=dict(
                    type="object",
                    properties={
                        "_argument_name1_": dict(
                            type="string", description="_argument_description1_"
                        ),
                        "_argument_name2_": dict(
                            type="string",
                            enum=["_argument_enum1_", "_argument_enum2_"],
                        ),
                    },
                    required=["_argument_name1_", "_argument_name2_"],
                ),
            ),
        ),
        dict(
            type="function",
            function=dict(
                name="_function_name2_",
                description="_function_description2_",
                parameters=dict(
                    type="object",
                    properties={
                        "_argument_name3_": dict(
                            type="array",
                            items=dict(type="string"),
                            description="_argument_description3_",
                        ),
                        "_argument_name4_": dict(
                            type="object",
                            description="_argument_description4_",
                            properties={
                                "_argument_name5_": dict(
                                    type="string",
                                    description="_argument_description5_",
                                )
                            },
                            required=["_argument_name5_"],
                        ),
                        "_argument_name6_": dict(
                            type="string",
                            enum=["_argument_enum3_", "_argument_enum4_"],
                            default="_argument_enum3_",
                        ),
                    },
                    required=["_argument_name3_", "_argument_name4_"],
                ),
            ),
        ),
        dict(
            type="function",
            function=dict(
                name="_function_name3_",
                parameters=dict(
                    type="object",
                    properties={
                        "_argument_number_": dict(type="number"),
                        "_argument_integer_": dict(type="integer"),
                        "_argument_boolean_": dict(type="boolean"),
                        "_argument_null_": dict(type="null"),
                    },
                ),
            ),
        ),
    ],
)


def test_consistency_with_tokenizer(response_template):
    """Check native rendering and prefix round trips at semantic marker boundaries."""
    tokenizer = build_tokenizer(response_template.name_or_path)
    tokenizer_messages = deepcopy(template_test_trajectory["messages"])
    for message in tokenizer_messages:
        if message["role"] == "assistant":
            message["reasoning_content"] = message["reasoning"]
            for tool_call in message.get("tool_calls", []):
                tool_call["function"]["arguments"] = json.loads(
                    tool_call["function"]["arguments"]
                )
    chatml_text = tokenizer.apply_chat_template(
        tokenizer_messages,
        tools=template_test_trajectory["tools"],
        tokenize=False,
        add_generation_prompt=False,
    )
    # The second assistant reply is after the last user query, so its reasoning is retained.
    templated_text = response_template.apply(template_test_trajectory["messages"][4])[
        "templated_prompt"
    ]
    assert templated_text in chatml_text

    # IDs may be omitted, and boolean/null spellings vary across native templates.
    markers = [
        "_assistant_reasoning2_",
        "_assistant_content2_",
        "_function_name1_",
        "_argument_name1_",
        "_argument_value1_",
        "_argument_name2_",
        "_argument_enum1_",
        "_function_name2_",
        "_argument_name3_",
        "_argument_value2_",
        "_argument_name4_",
        "_argument_name5_",
        "_argument_value3_",
        "_argument_name6_",
        "_argument_enum3_",
        "_function_name3_",
        "_argument_number_",
        "2609.24983",
        "_argument_integer_",
        "2609",
        "_argument_boolean_",
        "_argument_null_",
    ]
    marker_positions = []
    for marker in markers:
        assert marker in templated_text, f"Missing marker: {marker!r}"
        start = templated_text.rindex(marker)
        marker_positions.append((marker, start, start + len(marker)))

    for marker, start, end in marker_positions:
        for cut in (start, end):
            partial_text = templated_text[:cut]
            parsed_message = response_template.parse(
                partial_text,
                messages=template_test_trajectory["messages"][:4],
                tools=template_test_trajectory["tools"],
            )
            reapplied_text = response_template.apply(parsed_message)["templated_prompt"]
            assert partial_text.startswith(reapplied_text), (
                f"{marker!r} at {cut}:\n"
                f"original: {partial_text!r}\n"
                f"reapplied: {reapplied_text!r}"
            )
            # Prefix compatibility alone would let a parser silently discard whole channels.
            for completed_marker, _, completed_end in marker_positions:
                if completed_end <= cut:
                    assert (
                        len(reapplied_text) >= completed_end
                    ), f"{marker!r} at {cut}: lost completed marker {completed_marker!r}"


if __name__ == "__main__":
    with mximport.inpkg():
        from . import RESPONSE_TEMPLATE_CLASSES, CorrectionCallResponseTemplate

    for template_class in RESPONSE_TEMPLATE_CLASSES:
        # This serialization format has no corresponding native model tokenizer.
        if template_class is CorrectionCallResponseTemplate:
            continue
        response_template = template_class()
        test_consistency_with_tokenizer(response_template)
        print("test_consistency_with_tokenizer passed:", response_template.name_or_path)
