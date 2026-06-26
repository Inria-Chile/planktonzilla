"""
(c) Inria

Rich-based helpers for pretty-printing the Hydra config tree, enforcing run tags, and
rendering docstrings as Markdown in the terminal.
"""

import sys
from pathlib import Path
from typing import Any

import rich
import rich.markdown
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

# from pytorch_lightning.utilities import rank_zero_only
from rich.prompt import Prompt

from planktonzilla.utils.logger import get_pylogger

log = get_pylogger(__name__)


# @rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Print a DictConfig as a Rich tree, with fields ordered alphabetically by key.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        resolve (bool, optional): Whether to resolve interpolated reference fields of the DictConfig.
        save_to_file (bool, optional): Whether to also export the tree to ``config_tree.log`` in
            ``cfg.paths.output_dir``.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields to queue in sorted (alphabetical) order
    for field in sorted(cfg.keys()):
        (
            queue.append(field)
            if field in cfg
            else log.warning(f"Field '{field}' not found in config. Skipping '{field}' config printing...")
        )

    # add any remaining fields not already queued (defensive; sorted pass above covers all keys)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


# @rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config."""

    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)


# @rank_zero_only
def print_docstr_as_markdown(instance: Any) -> None:
    """Render ``instance``'s class name and dedented docstring as Markdown to the terminal."""

    def trim(docstring):
        """Code based on https://peps.python.org/pep-0257/#handling-docstring-indentation"""
        if not docstring:
            return ""
        # Convert tabs to spaces (following the normal Python rules)
        # and split into a list of lines:
        lines = docstring.expandtabs().splitlines()
        # Determine minimum indentation (first line doesn't count):
        indent = sys.maxsize
        for line in lines[1:]:
            stripped = line.lstrip()
            if stripped:
                indent = min(indent, len(line) - len(stripped))
        # Remove indentation (first line is special):
        trimmed = [lines[0].strip()]
        if indent < sys.maxsize:
            trimmed = trimmed + [line[indent:].rstrip() for line in lines[1:]]
        # Strip off trailing and leading blank lines:
        while trimmed and not trimmed[-1]:
            trimmed.pop()
        while trimmed and not trimmed[0]:
            trimmed.pop(0)
        # Return a single string:
        return "\n".join(trimmed)

    rich.print(rich.markdown.Markdown(f"# {instance.__class__.__name__}\n\n" + trim(instance.__doc__)))


if __name__ == "__main__":
    from hydra import compose, initialize

    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=False, overrides=[])
        print_config_tree(cfg, resolve=False, save_to_file=False)
