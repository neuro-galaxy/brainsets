#!/usr/bin/env python3
"""Discover changed OpenNeuro pipelines for PR smoke testing.

This script:
1. Finds changed files in brainsets_pipelines/*/pipeline.py
2. Parses each with AST (no imports) to detect OpenNeuro*Pipeline subclasses
3. Extracts brainset_id and optional ci_smoke_session
4. Outputs JSON matrix for GitHub Actions
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Optional


def get_name_from_base(base_node: ast.expr) -> Optional[str]:
    """Extract class name from base node (handles Name and Attribute)."""
    if isinstance(base_node, ast.Name):
        return base_node.id
    if isinstance(base_node, ast.Attribute):
        return base_node.attr
    return None


def extract_class_attributes(class_node: ast.ClassDef) -> dict:
    """Extract brainset_id and ci_smoke_session from class body."""
    attrs = {"brainset_id": None, "ci_smoke_session": None}

    for class_stmt in class_node.body:
        if not isinstance(class_stmt, ast.Assign):
            continue
        if len(class_stmt.targets) != 1:
            continue
        if not isinstance(class_stmt.targets[0], ast.Name):
            continue

        attr_name = class_stmt.targets[0].id
        if attr_name not in attrs:
            continue

        if isinstance(class_stmt.value, ast.Constant) and isinstance(
            class_stmt.value.value, str
        ):
            attrs[attr_name] = class_stmt.value.value

    return attrs


def discover_openneuro_pipelines(pipeline_files: list[Path]) -> list[dict]:
    """Discover OpenNeuro pipeline entries from changed files.

    Args:
        pipeline_files: List of pipeline.py file paths to parse

    Returns:
        List of dicts with keys: brainset_id, ci_smoke_session (optional)
    """
    openneuro_bases = {
        "OpenNeuroPipeline",
    }
    entries = []

    for pipeline_file in pipeline_files:
        try:
            module = ast.parse(
                pipeline_file.read_text(encoding="utf-8"),
                filename=str(pipeline_file),
            )
        except Exception as e:
            print(f"Error parsing {pipeline_file}: {e}", file=sys.stderr)
            continue

        for node in module.body:
            if not isinstance(node, ast.ClassDef):
                continue

            base_names = {get_name_from_base(base) for base in node.bases}
            if base_names.isdisjoint(openneuro_bases):
                continue

            attrs = extract_class_attributes(node)
            if attrs["brainset_id"] is None:
                print(
                    f"Warning: {pipeline_file} class {node.name} "
                    "missing brainset_id",
                    file=sys.stderr,
                )
                continue

            entry = {"brainset_id": attrs["brainset_id"]}
            if attrs["ci_smoke_session"]:
                entry["ci_smoke_session"] = attrs["ci_smoke_session"]

            entries.append(entry)

    return entries


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Discover changed OpenNeuro pipelines for CI smoke tests."
    )
    parser.add_argument(
        "pipeline_files",
        nargs="+",
        type=Path,
        help="One or more changed pipeline.py files to inspect.",
    )
    args = parser.parse_args()

    pipeline_files = args.pipeline_files
    entries = discover_openneuro_pipelines(pipeline_files)

    # Output as JSON for GitHub Actions matrix
    matrix = {"include": entries} if entries else {"include": []}
    print(json.dumps(matrix))


if __name__ == "__main__":
    main()
