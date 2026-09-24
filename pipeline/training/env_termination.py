# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Environment episode boundary for the pinned TRL tool loop.

Derived from TRL 1.6.0 GRPOTrainer._tool_call_loop. The native generation,
masking and log-probability bookkeeping stay here unchanged except for the
explicit episode-end guards. Like evaluation, malformed tool arguments become
feedback, while exceptions inside a tool abort instead of scoring broken
infrastructure. Native-source drift fails before the first loop.
Terminal feedback remains in the audit messages but is never fed to the model.
"""

import asyncio
import hashlib
import inspect
import textwrap

NATIVE_TOOL_LOOP_SHA256 = (
    "110682ce53ac985ff0eaae7b1a23f0026b5fd5e4f4d4e916686ba2a42f7f1133"
)
EPISODE_BOUNDARY = "env_done_or_budget_v1"


def verify_native_tool_loop(method):
    source = textwrap.dedent(inspect.getsource(method))
    if hashlib.sha256(source.encode()).hexdigest() != NATIVE_TOOL_LOOP_SHA256:
        raise RuntimeError(
            "Native TRL tool loop changed; requalify episode termination"
        )


def environment_tool_call_loop(
    self,
    prompts,
    prompt_ids,
    completion_ids,
    completions,
    logprobs,
    images,
    multimodal_fields,
    *,
    parse_response,
):
    if self._is_vlm or any(self._async_tool_dicts):
        raise ValueError("Environment termination requires synchronous text tools")
    if len(self.environments) != len(completions):
        raise ValueError("Environment slots must match the completion batch")
    # Tool execution loop: execute tools, then regenerate completions with tool results appended to the prompt
    tool_calls = [completion[0].get("tool_calls") for completion in completions]
    idxs_with_tool = [idx for idx, tool_call in enumerate(tool_calls) if tool_call]
    tool_calls = [tool_calls[idx] for idx in idxs_with_tool]
    tool_mask = [
        [1] * len(ids) for ids in completion_ids
    ]  # 0 for tool result tokens, 1 elsewhere
    # Collect images from multimodal tool responses for the forward pass
    tool_images = [[] for _ in completion_ids]
    tool_call_count = 0
    tool_failure_count = 0
    iteration_num = 0

    while idxs_with_tool and iteration_num < self.max_tool_calling_iterations:
        prompt_completion_tools = [
            prompts[i] for i in idxs_with_tool
        ]  # select only prompts that need tool calls
        # Snapshot state so we can rollback tool results that would exceed max_completion_length
        completions_len_before = [len(completions[i]) for i in idxs_with_tool]
        tool_images_len_before = [len(tool_images[i]) for i in idxs_with_tool]
        prompts_len_before = [len(prompts[i]) for i in idxs_with_tool]

        # Call the tools, and build the new prompt for generation
        for idx in range(len(idxs_with_tool)):
            idx_with_tool = idxs_with_tool[idx]
            tool_call_list = tool_calls[idx]
            prompt_completion_tool = prompt_completion_tools[idx]
            sync_tool_dict = self._sync_tool_dicts[idx_with_tool]
            async_tool_dict = self._async_tool_dicts[idx_with_tool]
            # Append the last assistant message (which triggered tool_calls) to the prompt
            prompt_completion_tool.append(completions[idx_with_tool][-1])
            async_coros = []
            tool_call_results = []
            for tool_call in tool_call_list:
                if self.environments[idx_with_tool].done:
                    break
                tool_call_count += 1
                if tool_call["type"] == "function":
                    function = tool_call["function"]
                    name = function["name"]
                    try:
                        if name in sync_tool_dict:
                            tool = sync_tool_dict[name]
                        elif name in async_tool_dict:
                            tool = async_tool_dict[name]
                        else:
                            raise ValueError(f"Tool {name} not found.")
                        inspect.signature(tool).bind(**function["arguments"])
                    except (TypeError, ValueError) as e:
                        tool_failure_count += 1
                        result = {"error": str(e)}
                        tool_call_results.append((name, result))
                    else:
                        if name in sync_tool_dict:
                            tool_call_results.append(
                                (name, tool(**function["arguments"]))
                            )
                        else:
                            async_coros.append((name, tool(**function["arguments"])))
                else:
                    tool_failure_count += 1
                    name = tool_call.get("name", "unknown")
                    tool_call_results.append(
                        (
                            name,
                            {
                                "error": f"Unsupported tool call type: {tool_call['type']}"
                            },
                        )
                    )

            if async_coros:

                async def _run_async_tools(async_coros):
                    coros = [coro for _, coro in async_coros]
                    results = await asyncio.gather(*coros, return_exceptions=True)
                    return [
                        (name, result)
                        for (name, _), result in zip(async_coros, results, strict=False)
                    ]

                async_results = asyncio.run_coroutine_threadsafe(
                    _run_async_tools(async_coros), self.async_loop
                ).result()

                for name, result in async_results:
                    if isinstance(result, Exception):
                        tool_failure_count += 1
                        tool_call_results.append((name, {"error": str(result)}))
                    else:
                        tool_call_results.append((name, result))

            for name, result in tool_call_results:
                # Support multimodal tool responses: if the tool returns a list of content blocks
                # (e.g., [{"type": "image", "image": ...}, {"type": "text", "text": "..."}]),
                # pass them through directly so _tokenize_prompts can extract images for VLMs.
                content = result if isinstance(result, list) else str(result)
                tool_message = {"role": "tool", "name": name, "content": content}
                # Collect images from multimodal tool responses
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "image":
                            tool_images[idx_with_tool].append(part["image"])
                prompt_completion_tool.append(tool_message)
                completions[idx_with_tool].append(tool_message)

        # Stop at the terminal action or last permitted assistant turn. Retain
        # feedback for the audit/rewards, but no later model context is needed:
        # terminal feedback and an unused generation prefix have no loss tokens.
        continuing = [
            j
            for j, i in enumerate(idxs_with_tool)
            if not self.environments[i].done
            and iteration_num + 1 < self.max_tool_calling_iterations
        ]
        idxs_with_tool = [idxs_with_tool[j] for j in continuing]
        completions_len_before = [completions_len_before[j] for j in continuing]
        tool_images_len_before = [tool_images_len_before[j] for j in continuing]
        prompts_len_before = [prompts_len_before[j] for j in continuing]
        if not idxs_with_tool:
            break

        # Build token IDs by concatenation: prompt + completion + tool_suffix.
        prompt_completion_tool_ids = []
        for idx in range(len(idxs_with_tool)):
            idx_with_tool = idxs_with_tool[idx]
            # Extract trailing tool messages from completions
            tool_messages = []
            for message in reversed(completions[idx_with_tool]):
                if message["role"] == "tool":
                    tool_messages.insert(0, message)
                else:
                    break
            suffix_ids = self._get_tool_suffix_ids(tool_messages)
            prompt_completion_tool_ids.append(
                prompt_ids[idx_with_tool] + completion_ids[idx_with_tool] + suffix_ids
            )

        # Drop tool results whose addition would push the sequence past max_completion_length (the completion
        # budget) or past the backend context ceiling (vLLM and transformers will error out on inputs longer than
        # the model's max length). The sample exits the loop with its completion as-is, and the tool
        # messages/images appended this iteration are rolled back so completions and tool_images stay consistent
        # with completion_ids.
        if self.use_vllm and self.vllm_mode == "colocate":
            max_model_len = (
                self.vllm_generation.llm.llm_engine.model_config.max_model_len
            )
        else:
            config = (
                self.model.config.text_config if self._is_vlm else self.model.config
            )
            max_model_len = config.max_position_embeddings
        overlong = [
            len(pct) - len(prompt_ids[i]) > self.max_completion_length
            or len(pct) >= max_model_len
            for i, pct in zip(idxs_with_tool, prompt_completion_tool_ids, strict=True)
        ]
        for idx in range(len(idxs_with_tool)):
            if overlong[idx]:
                idx_with_tool = idxs_with_tool[idx]
                del completions[idx_with_tool][completions_len_before[idx] :]
                del tool_images[idx_with_tool][tool_images_len_before[idx] :]
                del prompts[idx_with_tool][prompts_len_before[idx] :]
        # Keep only non-overlong items for further processing
        idxs_with_tool = [
            idx for idx, o in zip(idxs_with_tool, overlong, strict=True) if not o
        ]
        prompt_completion_tool_ids = [
            pct
            for pct, o in zip(prompt_completion_tool_ids, overlong, strict=True)
            if not o
        ]
        if not idxs_with_tool:
            break  # all overlong, exit tool loop

        # Filter images and multimodal fields to match the current subset (index into full batch).
        # Merge tool response images so the model can see visual feedback during generation.
        merged_images = images
        if any(imgs for imgs in tool_images):
            if merged_images is None:
                merged_images = [imgs if imgs else None for imgs in tool_images]
            else:
                merged_images = [
                    (existing or []) + new
                    for existing, new in zip(merged_images, tool_images, strict=True)
                ]
        loop_images = (
            [merged_images[i] for i in idxs_with_tool] if merged_images else None
        )
        if multimodal_fields:
            loop_multimodal_fields = {}
            for k, v in multimodal_fields.items():
                selected = [v[i] for i in idxs_with_tool]
                # Per-token fields (e.g. token_type_ids) need zero-padding to match extended prompt length
                if isinstance(selected[0], list):
                    selected = [
                        s + [0] * (len(pct) - len(s))
                        for s, pct in zip(
                            selected, prompt_completion_tool_ids, strict=True
                        )
                    ]
                loop_multimodal_fields[k] = selected
        else:
            loop_multimodal_fields = {}

        # Generate new completions after tool execution (using concatenated IDs, no re-tokenization)
        post_tool_ids, post_tool_logprobs = self._generate_single_turn(
            prompt_completion_tool_ids, loop_images, loop_multimodal_fields
        )

        # Truncate so that pct[len(prompt_ids[idx]) :] + post_tool does not exceed max_completion_length.
        # The pre-regen check guarantees len(completion_tool_ids) <= max_completion_length, so any
        # excess can only come from post_tool_ids. post_tool_ids is model-generated text and never
        # contains image tokens, so a plain slice is safe.
        for idx in range(len(idxs_with_tool)):
            idx_with_tool = idxs_with_tool[idx]
            completion_tool_length = len(prompt_completion_tool_ids[idx]) - len(
                prompt_ids[idx_with_tool]
            )
            excess_length = (
                completion_tool_length
                + len(post_tool_ids[idx])
                - self.max_completion_length
            )
            if excess_length > 0:
                new_len = len(post_tool_ids[idx]) - excess_length
                post_tool_ids[idx] = post_tool_ids[idx][:new_len]
                if logprobs is not None:
                    post_tool_logprobs[idx] = post_tool_logprobs[idx][:new_len]

        # Update tool_mask: the tool result should be 0 and the post-tool 1
        for idx in range(len(idxs_with_tool)):
            idx_with_tool = idxs_with_tool[idx]
            prompt_completion_tool_length = len(prompt_completion_tool_ids[idx])
            prompt_length = len(prompt_ids[idx_with_tool])
            completion_length = len(completion_ids[idx_with_tool])
            post_tool_length = len(post_tool_ids[idx])
            tool_length = (
                prompt_completion_tool_length - prompt_length - completion_length
            )
            tool_mask[idx_with_tool] += [0] * tool_length + [1] * post_tool_length
            if logprobs is not None:
                logprobs[idx_with_tool] += [0.0] * tool_length + post_tool_logprobs[idx]

        # Update completion_ids with the new completions (after tool execution)
        for idx in range(len(idxs_with_tool)):
            idx_with_tool = idxs_with_tool[idx]
            prompt_length = len(prompt_ids[idx_with_tool])
            pct = prompt_completion_tool_ids[idx]  # = prompt-completion-tool
            completion_ids[idx_with_tool] = pct[prompt_length:] + post_tool_ids[idx]

        # Decode post-tool completions.
        post_tool_completions = [
            parse_response(self._tokenizer, ids) if ids else {} for ids in post_tool_ids
        ]

        # Add post-tool completions to the existing completions
        for idx in range(len(idxs_with_tool)):
            idx_with_tool = idxs_with_tool[idx]
            if post_tool_completions[
                idx
            ]:  # {} if post-tool completions completely truncated
                completions[idx_with_tool].append(post_tool_completions[idx])

        # Check for further tool calls
        tool_calls = [
            completion.get("tool_calls") for completion in post_tool_completions
        ]
        idxs_with_tool = [
            idx
            for idx, tool_call in zip(idxs_with_tool, tool_calls, strict=True)
            if tool_call
        ]
        tool_calls = [tool_call for tool_call in tool_calls if tool_call]
        iteration_num += 1

    return (
        tool_mask,
        completions,
        completion_ids,
        logprobs,
        tool_call_count,
        tool_failure_count,
        tool_images,
    )
