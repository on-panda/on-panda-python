# Develop Document

## TODO
- [ ] token level supervision
    - [ ] EOT token `<|stop|>`
    - continue_with chosen VS input
    - two type of rejected token's loss
        1. $log(p_{rej})$
        2. $-log(1-p_{rej})$


## DONE


## Iterative Correction: One Round (Proposed Design)

The following diagram specifies one correction round. `CM` is the correcting model,
`FAR` is its find-and-replace correction, `CM-RT` is the correcting model's response
template, and `Policy-RT` is the policy model's response template. The two templates may
be different: `CM-RT` is used to locate and apply the correction, while `Policy-RT` is
used to continue the policy response.

The policy response is first rendered in `CM-RT` space, preserving reasoning and tool-call
markers, to produce the rejected CM rendering. The correcting model returns a FAR
correction; applying it to that rendering produces a CM continuation prefix. Parsing the
prefix yields the structured `partial_messages`, which are rendered with `Policy-RT` to
obtain a full policy continuation prefix. The prefix is token-aligned and truncated to
produce the policy prefix used to request and continue the policy response. Parsing that
response back with `Policy-RT` produces `corrected_messages`, which can seed a later
correction round.

```mermaid
flowchart TB
    policy_messages["Policy messages"]
    cm_messages["CM request messages"]
    far["FAR correction"]
    rejected_cm_templated["Rejected response<br/>(CM-RT rendered)"]
    cm_continue_prefix["CM continuation prefix"]
    partial_messages["Partially corrected messages<br/>(partial_messages)"]
    prefix_full["Full policy continuation prefix"]
    prefix_truncated["Token-aligned policy prefix<br/>(truncated)"]
    policy_templated_response["Policy templated response"]
    corrected_messages["Corrected policy messages<br/>(corrected_messages)"]
    special_tokens["Template markers<br/>&lt;|reasoning|&gt;: reasoning marker<br/>&lt;|tool_calls|&gt;: default tool-call marker"]

    policy_messages -->|"append CM system prompt"| cm_messages
    cm_messages -->|"request correction from CM"| far
    policy_messages -->|"apply CM-RT<br/>(preserve reasoning/tool-call markers)"| rejected_cm_templated
    far -->|"correction payload"| cm_continue_prefix
    rejected_cm_templated -->|"apply FAR correction"| cm_continue_prefix
    cm_continue_prefix -->|"parse with CM-RT"| partial_messages
    partial_messages -->|"apply Policy-RT"| prefix_full
    prefix_full -->|"token-align and truncate"| prefix_truncated
    prefix_truncated -->|"request & continue"| policy_templated_response
    policy_templated_response -->|"parse with Policy-RT"| corrected_messages

    special_tokens -.-> rejected_cm_templated
```
