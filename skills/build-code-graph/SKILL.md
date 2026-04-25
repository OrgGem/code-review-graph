---
name: build-code-graph
description: Build or update the code graph for Claude, Kiro, and Antigravity workflows.
argument-hint: "[full]"
---

# Build Code Graph

Build or incrementally update the persistent code knowledge graph for this repository.

## Steps

1. **Check graph status** with `list_graph_stats_tool`.
   - If the graph has never been built (`last_updated` is null), use a full rebuild.
   - Otherwise, use incremental update.

2. **Build the graph** with `build_or_update_graph_tool`:
   - First-time setup: `build_or_update_graph_tool(full_rebuild=True)`
   - Regular updates: `build_or_update_graph_tool()`

3. **Verify results** with `list_graph_stats_tool`:
   - Files parsed
   - Nodes and edges created
   - Languages detected
   - Any parsing errors

## When to Use

- First-time setup in a repository
- After major refactors or branch switches
- When graph data is stale
- Before review/debug/refactor tasks in Claude, Kiro, or Antigravity

## Notes

- Graph database: `.code-review-graph/graph.db`
- Incremental hooks keep the graph updated after edits/commits
- Binary/generated files and `.code-review-graphignore` patterns are skipped
