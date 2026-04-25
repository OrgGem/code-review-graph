from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

DEFAULT_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".swift",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".scala",
    ".sql",
    ".md",
    ".json",
    ".yaml",
    ".yml",
}

IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".code-review-graph",
    "dist",
    "build",
    "__pycache__",
}


@dataclass
class Chunk:
    path: str
    start_line: int
    end_line: int
    text: str


def iter_files(project_root: Path) -> list[Path]:
    files: list[Path] = []
    for root, dirs, filenames in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        root_path = Path(root)
        for filename in filenames:
            path = root_path / filename
            if path.suffix.lower() in DEFAULT_EXTENSIONS:
                files.append(path)
    return files


def split_chunks(path: Path, content: str, max_chars: int = 1500, overlap: int = 200) -> list[Chunk]:
    lines = content.splitlines()
    chunks: list[Chunk] = []
    i = 0

    while i < len(lines):
        start = i
        size = 0
        while i < len(lines):
            line_len = len(lines[i]) + 1
            if size + line_len > max_chars and i > start:
                break
            size += line_len
            i += 1

        text = "\n".join(lines[start:i]).strip()
        if text:
            chunks.append(
                Chunk(
                    path=str(path),
                    start_line=start + 1,
                    end_line=i,
                    text=text,
                )
            )

        if i >= len(lines):
            break

        chars_back = 0
        j = i - 1
        while j >= 0 and chars_back < overlap:
            chars_back += len(lines[j]) + 1
            j -= 1
        i = max(j + 1, start + 1)

    return chunks


def hash_embed(text: str, dim: int = 384) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text.lower())
    for token in tokens:
        idx = hash(token) % dim
        vec[idx] += 1.0

    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def build_index(project_root: Path, output_dir: Path, dim: int = 384) -> tuple[int, int]:
    files = iter_files(project_root)
    all_chunks: list[Chunk] = []

    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        all_chunks.extend(split_chunks(path, text))

    if not all_chunks:
        raise RuntimeError("No chunks found. Check project root and supported file extensions.")

    vectors = np.stack([hash_embed(chunk.text, dim=dim) for chunk in all_chunks]).astype(np.float32)

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "code.index"))

    metadata = [
        {
            "id": i,
            "path": chunk.path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "preview": chunk.text[:300],
        }
        for i, chunk in enumerate(all_chunks)
    ]
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dimension": dim,
                "file_count": len(files),
                "chunk_count": len(all_chunks),
                "chunks": metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return len(files), len(all_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS code index for a project")
    parser.add_argument("--project-root", default=".", help="Path to project root")
    parser.add_argument(
        "--output-dir",
        default=".code-review-graph/faiss-index",
        help="Output directory for FAISS index and metadata",
    )
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()

    file_count, chunk_count = build_index(project_root=project_root, output_dir=output_dir, dim=args.dim)
    print(f"FAISS index built: files={file_count}, chunks={chunk_count}, output={output_dir}")


if __name__ == "__main__":
    main()
