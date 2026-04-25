---
name: build-code-graph
description: Build code graph, initialize MCP config, and create a FAISS index for repository code search.
argument-hint: "[project_root]"
---

# Build Code Graph (FAISS + MCP)

This skill initializes MCP configuration and builds a local FAISS index for source code in one project.

## Prerequisites

1. Python 3.10+
2. Install dependencies:

```bash
python3 -m pip install faiss-cpu numpy
python3 -m pip install code-review-graph
```

## Steps

1. **Build/update code graph**
   - Run:
   ```bash
   code-review-graph build
   ```

2. **Initialize MCP config (template)**
   - Copy `skill-build/build-code-graph/mcp.config.template.json` to your MCP client config
   - Keep `code-review-graph` entry for structural graph tools
   - Add `faiss-code-index` entry for semantic chunk search

3. **Build FAISS index**
   - Run:
   ```bash
   python3 skill-build/build-code-graph/build_faiss_index.py --project-root .
   ```

4. **Run FAISS MCP server**
   - Run:
   ```bash
   python3 skill-build/build-code-graph/faiss_mcp_server.py --index-dir .code-review-graph/faiss-index
   ```

## Output

- `.code-review-graph/faiss-index/code.index` (FAISS index)
- `.code-review-graph/faiss-index/metadata.json` (chunk metadata)

## Notes

- The FAISS index is local-only and never uploads source code.
- Re-run the build script after large code changes.
