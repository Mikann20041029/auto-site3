from __future__ import annotations
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import Any
from .util import write_text

def env_for(templates_dir: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"])
    )

def write_asset(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())

def render_to_file(jenv: Environment, template_name: str, context: dict[str, Any], out_path: Path) -> None:
    html = jenv.get_template(template_name).render(**context)
    write_text(out_path, html)
