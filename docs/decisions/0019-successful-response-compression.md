# Compare successful-response costs, then compress the competent E1 policy

Decided: 2026-09-23. The user approved comparing the linear and relative
successful-response length costs on saved batches, followed by a controlled
compression experiment starting from the competent E1 policy. The user
explicitly declined additional gradient-history preservation.

Keep the completed cosine comparison and its adverse result unchanged. Its
protocol remains the authority for that result. For the next experiment,
compare bounded costs on successful responses; failed responses receive no
length signal. Retain the assistant-token ruler, environment completion
boundary, task reward, mean-only advantages, model, and installed stack.

Choose the candidate using training/development data and declared mechanical
criteria before new training. Offline reward replay establishes incentives,
not learned compression. Then compare two copies of the same final E1 adapter:
task-only continuation and selected-cost continuation, with identical geometry,
optimizer initialization, horizon, data and evaluation. This asks whether an
already competent policy can compress, rather than repeating the from-base
experiment. No new gradient logging or gradient-history infrastructure is
authorized or needed.

The concrete procedure and limits are in
[the continuation plan](../plans/read-table-compression.md).
