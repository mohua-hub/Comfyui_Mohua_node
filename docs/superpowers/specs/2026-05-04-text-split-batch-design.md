# Text Split Batch Node Design

**Date:** 2026-05-04

## Goal

Add a single ComfyUI utility node that splits an input string into batched string outputs. The node must support two user-selectable modes:

- Split by delimiter
- Split by fixed character length

The node must return the split text as a `STRING` list output and also return the total number of non-empty segments.

## Context

The current `nodes/tools.py` file already contains utility-style nodes such as `LoadImagesMulti` and `ProcessString`. A `TextSplitBatch` class stub exists but is not implemented or registered in `__init__.py`.

This work should follow the existing lightweight utility-node style in this repository and avoid unrelated refactors.

## Node Shape

### Class name

`TextSplitBatch`

### Suggested registered node id

`TextSplitBatch_MohuaAI`

### Category

`Mohua_tools`

## Inputs

### Required

- `input_string`: `STRING`
  - Multiline enabled
  - Source text to split
- `split_mode`: enum
  - `delimiter`
  - `fixed_length`

### Optional behavior/config inputs

- `delimiter`: `STRING`
  - Used only when `split_mode == "delimiter"`
  - Default should be a newline so the node is useful immediately for line-based batching
- `delimiter_mode`: enum
  - `plain`
  - `regex`
  - `plain` is the default for safer general use
- `chunk_length`: `INT`
  - Used only when `split_mode == "fixed_length"`
  - Minimum value must be `1`
  - Default can be `1` or a small practical value such as `10`
- `trim_result`: `BOOLEAN`
  - Default `True`
  - If enabled, trim leading and trailing whitespace from each segment before filtering/output
- `drop_empty`: `BOOLEAN`
  - Default `True`
  - If enabled, discard empty segments after optional trimming

## Outputs

- `text_list`: `STRING`
  - Batched/list output
- `count`: `INT`
  - Number of segments returned in `text_list`

The node should mark only the first output as list output via `OUTPUT_IS_LIST`.

## Behavior

## Mode 1: Delimiter Split

- If `delimiter_mode == "plain"`, use literal string splitting.
- If `delimiter_mode == "regex"`, use `re.split`.
- If the delimiter is empty in delimiter mode, do not raise an exception.
  - Instead, return the original input as a single segment when non-empty after filtering.
  - This keeps the node forgiving in ComfyUI workflows.
- After splitting:
  - If `trim_result` is enabled, run `strip()` on each segment.
  - If `drop_empty` is enabled, remove empty segments.

Examples:

- Input `a,b,,c`, delimiter `,`, trim on, drop empty on -> `["a", "b", "c"]`
- Input `a | b | c`, delimiter `|`, trim on -> `["a", "b", "c"]`

## Mode 2: Fixed Length Split

- Split by character count, not bytes.
- Preserve original order.
- The final chunk may be shorter than `chunk_length`.
- `chunk_length <= 0` is invalid and must raise a clear `ValueError`.
- After chunking:
  - If `trim_result` is enabled, trim each chunk.
  - If `drop_empty` is enabled, discard empty results.

Example:

- Input `abcdefg`, length `3` -> `["abc", "def", "g"]`

## Empty Input Rules

- Empty input should return:
  - `text_list = []`
  - `count = 0`

This keeps downstream batch consumers predictable.

## Error Handling

- Invalid `split_mode`: raise `ValueError`
- Fixed-length mode with `chunk_length <= 0`: raise `ValueError`
- Regex delimiter errors from `re.split` should surface clearly rather than being silently swallowed

## Registration

Update `__init__.py` so the new node appears in:

- `NODE_CLASS_MAPPINGS`
- `NODE_DISPLAY_NAME_MAPPINGS`

No other node registrations should be changed unless required for consistency.

## Implementation Notes

- Keep implementation self-contained in `nodes/tools.py`
- Prefer small helper methods inside `TextSplitBatch` for:
  - post-processing segments
  - delimiter splitting
  - fixed-length chunking
- Follow the repository's existing simple class-based ComfyUI node style
- Avoid changing existing `ProcessString` behavior as part of this task

## Verification

Minimum verification should cover:

- Delimiter split with plain delimiter
- Delimiter split with repeated delimiters and empty entries removed
- Fixed-length split with remainder chunk
- Empty input
- Count output matches returned list length
- Node registration is importable from package `__init__.py`

## Out of Scope

- Adding preview/combined text outputs
- Adding multi-delimiter presets beyond the provided delimiter input
- Refactoring the unrelated string-processing node
- Changing repo-wide encoding issues unless they block this feature directly
