"""Extract coupling reaction info (DOI, title, metal, support, bond type) from article JSONs via DeepSeek.

Supports multi-threaded parallel extraction for high throughput.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from json_repair import repair_json
from openai import OpenAI


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
SUPPORTED_MODEL = "deepseek-chat"
DEFAULT_OUTPUT = Path("analysis_results/coupling_info_results.json")
DEFAULT_CSV_OUTPUT = Path("analysis_results/coupling_info_results.csv")
DEFAULT_WORKERS = 4
PERSIST_EVERY = 5  # write to disk every N new records


# ---------------------------------------------------------------------------
# DeepSeek client
# ---------------------------------------------------------------------------

class DeepSeekClient:
    """Thin wrapper around the OpenAI SDK with retry logic."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEEPSEEK_BASE_URL,
        model: str = SUPPORTED_MODEL,
    ) -> None:
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DeepSeek API key missing. Set DEEPSEEK_API_KEY env var.")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> Tuple[str, Dict[str, int]]:
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    stream=False,
                )
                content = (response.choices[0].message.content or "").strip()
                usage = getattr(response, "usage", None)
                usage_dict: Dict[str, int] = {}
                if usage:
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        value = getattr(usage, key, None)
                        if isinstance(value, int):
                            usage_dict[key] = value
                return content, usage_dict
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))
        raise RuntimeError("DeepSeek request failed") from last_exc


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def parse_json_payload(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = repair_json(raw)
        return json.loads(fixed)


def load_article_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_article_files(root: Path) -> Iterable[Path]:
    for json_path in sorted(root.rglob("*.json")):
        if json_path.name.startswith("._"):
            continue
        yield json_path


def extract_compact_text(section: Any, max_chars: int = 4000) -> Optional[str]:
    """Recursively collect text from structured sections, truncated to *max_chars*.

    Prioritises Introduction / Results / Conclusion sections over Experimental.
    """
    PRIORITY_KEYWORDS = ("introduction", "result", "discussion", "conclusion", "overview", "summary")
    DEPRIORITY_KEYWORDS = ("experimental", "experiment section", "supporting info", "acknowledgement", "credit", "appendix")

    def _collect(obj: Any) -> str:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            pieces: List[str] = []
            for key in ("title", "text", "content", "footnote", "description"):
                value = obj.get(key)
                if isinstance(value, (str, int, float)):
                    pieces.append(str(value))
                elif value:
                    pieces.append(_collect(value))
            if not pieces:
                for value in obj.values():
                    pieces.append(_collect(value))
            return "\n".join(filter(None, pieces))
        if isinstance(obj, list):
            return "\n\n".join(filter(None, (_collect(item) for item in obj)))
        return str(obj)

    # If section is a list of dicts with "title", reorder to prioritise useful sections
    if isinstance(section, list):
        priority: List[Any] = []
        normal: List[Any] = []
        depriority: List[Any] = []
        for item in section:
            title_lower = ""
            if isinstance(item, dict):
                title_lower = str(item.get("title", "")).lower()
            if any(kw in title_lower for kw in PRIORITY_KEYWORDS):
                priority.append(item)
            elif any(kw in title_lower for kw in DEPRIORITY_KEYWORDS):
                depriority.append(item)
            else:
                normal.append(item)
        section = priority + normal + depriority

    text = _collect(section).strip()
    if not text:
        return None
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]..."
    return text


# ---------------------------------------------------------------------------
# Resume / persistence
# ---------------------------------------------------------------------------

def load_existing_results(output_path: Path) -> Tuple[List[Dict[str, Any]], Set[str]]:
    results: List[Dict[str, Any]] = []
    processed_keys: Set[str] = set()
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                data = json.loads(content)
                results = data if isinstance(data, list) else [data]
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in existing results: {output_path}") from exc
    for rec in results:
        key = rec.get("doi") or rec.get("source_path")
        if key:
            processed_keys.add(str(key))
    return results, processed_keys


def write_results(results: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def export_csv(results: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["doi", "title", "catalysis_type", "metal", "support", "ligand_name", "coupling_type"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in results:
            writer.writerow({
                "doi": rec.get("doi", ""),
                "title": rec.get("title", ""),
                "catalysis_type": rec.get("catalysis_type", ""),
                "metal": rec.get("metal", ""),
                "support": rec.get("support", ""),
                "ligand_name": rec.get("ligand_name", ""),
                "coupling_type": rec.get("coupling_type", ""),
            })


# ---------------------------------------------------------------------------
# Token usage tracker
# ---------------------------------------------------------------------------

class UsageAccumulator:
    """Thread-safe token usage aggregator."""

    def __init__(self) -> None:
        self._totals: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._lock = threading.Lock()

    def add(self, usage: Dict[str, int]) -> None:
        with self._lock:
            for key in self._totals:
                self._totals[key] += usage.get(key, 0)

    def as_dict(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._totals)


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

class ProgressBar:
    """Thread-safe text progress bar."""

    def __init__(self, total: int, quiet: bool = False) -> None:
        self._total = total
        self._quiet = quiet
        self._current = 0
        self._bar_width = 30
        self._lock = threading.Lock()

    def _render(self) -> str:
        denom = max(self._total, 1)
        frac = min(self._current / denom, 1.0)
        filled = int(self._bar_width * frac)
        bar = "#" * filled + "-" * (self._bar_width - filled)
        pct = int(frac * 100)
        return f"\rProgress: [{bar}] {pct:3d}% ({self._current}/{self._total})"

    def update(self) -> None:
        if self._quiet:
            return
        with self._lock:
            self._current += 1
            print(self._render(), end="", flush=True)

    def finish(self) -> None:
        if self._quiet:
            return
        with self._lock:
            print(self._render(), end="", flush=True)
            print()


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a catalysis expert. Given a research article (title, abstract, and optionally \
full text and tables), perform TWO tasks:

1. **Relevance check**: Determine whether the article describes a catalyst-mediated \
coupling reaction that forms C-C, C-O, or C-N bonds. Common coupling reactions include \
Suzuki, Heck, Sonogashira, Kumada, Negishi, Stille, Buchwald-Hartwig, Ullmann, Chan-Lam, \
and cross-dehydrogenative coupling (CDC). \
Mark as NOT relevant if:
  - The article focuses on oxidation, reduction, polymerization, HER/OER/CO2RR, \
material characterization only, adsorption, degradation, etc. WITHOUT a coupling reaction.
  - The "catalyst" is merely a metal electrode, metal foil, or metal vessel used in \
an electrochemical or mechanochemical setup and no actual catalytic material is designed \
for the coupling step. Pure electrochemical coupling driven by applied potential on \
a bare electrode does NOT count.

2. **Information extraction** (only if relevant): Extract the following fields:
   - `catalysis_type`: Must be exactly "homogeneous" or "heterogeneous".
     * "homogeneous" — catalyst is a soluble molecular complex (metal salt + ligand, \
organometallic complex, metal complex in solution, photoredox catalyst in solution).
     * "heterogeneous" — catalyst is a solid material (supported metal, nanoparticles, \
metal oxide, carbon material, etc.) that operates in a different phase from the reactants.
   - `metal`: The catalytic metal element(s). Use standard element symbols (e.g. "Pd", \
"Ni", "Cu", "Ru"). If multiple metals, separate with "/" (e.g. "Pd/Cu"). \
If no metal catalyst is used, return "none". Do NOT include support or ligand here.
   - `support`: **Only for heterogeneous catalysts.** Classify the solid support into \
exactly ONE category using these PRIORITY RULES (apply top to bottom, first match wins):
     1. "Nitride" — if the support contains nitrogen: g-C3N4, BN, carbon nitride, \
C3N4-based composites, BCN, etc.
     2. "Carbon" — if no nitrogen but support is carbon-based: activated carbon, \
graphene, graphene oxide, CNT, carbon black, biochar, carbon fiber, carbon sphere, etc.
     3. "Metal Oxides" — any oxide material: TiO2, Al2O3, ZrO2, CeO2, Fe3O4, SiO2, \
Cu2O, ZnO, Fe2O3, mixed metal oxides, spinel oxides, hydrotalcites, etc. \
If the material contains both metal and oxygen (e.g. Cu2O, Fe3O4, CuO@Pd), classify \
as "Metal Oxides".
     4. "Metals" — pure metallic nanoparticles, bimetallic alloys, or metal surfaces \
with NO oxide or other non-metal support (e.g. PdAu alloy NPs, nanoporous Au, Pd NPs \
without any support). Do NOT classify metal electrodes or metal vessels here.
     5. "Others" — zeolites, clays, polymers, or anything not fitting above.
     For homogeneous catalysts, set support to null.
   - `ligand_name`: **Only for homogeneous catalysts.** Give the key ligand or ligand \
class name (e.g. "PPh3", "NHC", "bipyridine", "dppf", "Sadphos", "BINAP"). \
If the ligand is not specified or not applicable, return null. \
For heterogeneous catalysts, set ligand_name to null.
   - `coupling_type`: The bond type formed. Must be one of: "C-C", "C-O", "C-N", or \
"unknown". If multiple types, separate with "/" (e.g. "C-O/C-N").

Return ONLY compact JSON. Examples:

Pd/C catalyzed Suzuki coupling:
{"relevant":true,"catalysis_type":"heterogeneous","metal":"Pd","support":"Carbon","ligand_name":null,"coupling_type":"C-C","rationale":"Pd on carbon support, heterogeneous Suzuki coupling."}

Pd on g-C3N4 photocatalytic coupling:
{"relevant":true,"catalysis_type":"heterogeneous","metal":"Pd","support":"Nitride","ligand_name":null,"coupling_type":"C-C","rationale":"Pd on g-C3N4 (nitrogen-containing), classified as Nitride."}

Cu2O@PdCu catalyzed C-H arylation:
{"relevant":true,"catalysis_type":"heterogeneous","metal":"Pd/Cu","support":"Metal Oxides","ligand_name":null,"coupling_type":"C-C","rationale":"Cu2O contains oxygen, so support is Metal Oxides."}

PdFe alloy NPs Hiyama coupling:
{"relevant":true,"catalysis_type":"heterogeneous","metal":"Pd/Fe","support":"Metals","ligand_name":null,"coupling_type":"C-C","rationale":"PdFe alloy NPs with no oxide or other support, classified as Metals."}

Pd/Al2O3 catalyzed Heck reaction:
{"relevant":true,"catalysis_type":"heterogeneous","metal":"Pd","support":"Metal Oxides","ligand_name":null,"coupling_type":"C-C","rationale":"Pd on alumina, heterogeneous Heck reaction."}

Pd/Sadphos asymmetric Heck reaction:
{"relevant":true,"catalysis_type":"homogeneous","metal":"Pd","support":null,"ligand_name":"Sadphos","coupling_type":"C-C","rationale":"Soluble Pd/Sadphos complex for homogeneous Heck coupling."}

Electrochemical coupling on bare Ni foam electrode:
{"relevant":false,"catalysis_type":null,"metal":null,"support":null,"ligand_name":null,"coupling_type":null,"rationale":"Bare metal electrode is not a designed catalyst; excluded."}

TiO2 photocatalytic water splitting:
{"relevant":false,"catalysis_type":null,"metal":null,"support":null,"ligand_name":null,"coupling_type":null,"rationale":"Water splitting is not a coupling reaction."}

Now analyze the following article:
"""


class CouplingInfoExtractor:
    """Single-call extractor: relevance + info in one LLM request."""

    def __init__(self, client: DeepSeekClient) -> None:
        self._client = client

    def extract(
        self,
        title: str,
        abstract: str,
        doi: Optional[str] = None,
        full_text: Optional[str] = None,
        tables: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        parts: List[str] = []
        if doi:
            parts.append(f"DOI: {doi}")
        if title:
            parts.append(f"Title: {title}")
        parts.append(f"Abstract:\n{abstract.strip()}")
        if full_text:
            parts.append(f"Full Text (excerpt):\n{full_text.strip()}")
        if tables:
            parts.append(f"Tables:\n{tables.strip()}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        raw, usage = self._client.complete(messages)
        payload = parse_json_payload(raw)
        return payload, usage


# ---------------------------------------------------------------------------
# Prepare tasks (pre-filter before submitting to thread pool)
# ---------------------------------------------------------------------------

def prepare_tasks(
    article_paths: Iterable[Path],
    processed_keys: Set[str],
    limit: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Load articles and pre-filter; return a list of task dicts ready for LLM calls."""
    tasks: List[Dict[str, Any]] = []
    for article_path in article_paths:
        if limit is not None and len(tasks) >= limit:
            break
        try:
            article = load_article_json(article_path)
        except Exception as exc:
            if verbose:
                print(f"\nSkip {article_path} (load error: {exc})")
            continue

        metadata = article.get("metadata", {})
        abstract = metadata.get("abstract")
        if not abstract:
            continue

        doi = metadata.get("doi")
        title = metadata.get("title", "")
        identifier = doi or str(article_path)
        if identifier in processed_keys:
            continue

        full_text_str = extract_compact_text(article.get("full_text"))
        tables_str = extract_compact_text(article.get("tables"))

        tasks.append({
            "article_path": article_path,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "full_text": full_text_str,
            "tables": tables_str,
            "identifier": identifier,
        })
    return tasks


# ---------------------------------------------------------------------------
# Single-article worker (runs inside thread pool)
# ---------------------------------------------------------------------------

def _process_one(
    task: Dict[str, Any],
    extractor: CouplingInfoExtractor,
    usage_acc: UsageAccumulator,
) -> Optional[Dict[str, Any]]:
    """Process one article. Returns a record dict or None."""
    try:
        payload, usage = extractor.extract(
            title=task["title"],
            abstract=task["abstract"],
            doi=task["doi"],
            full_text=task["full_text"],
            tables=task["tables"],
        )
        usage_acc.add(usage)
    except Exception:
        return None

    if not bool(payload.get("relevant")):
        return None

    return {
        "doi": task["doi"],
        "title": task["title"],
        "source_path": str(task["article_path"]),
        "catalysis_type": payload.get("catalysis_type"),
        "metal": payload.get("metal"),
        "support": payload.get("support"),
        "ligand_name": payload.get("ligand_name"),
        "coupling_type": payload.get("coupling_type"),
        "rationale": payload.get("rationale"),
    }


# ---------------------------------------------------------------------------
# Multi-threaded pipeline
# ---------------------------------------------------------------------------

def analyze_articles(
    tasks: List[Dict[str, Any]],
    extractor: CouplingInfoExtractor,
    existing_results: List[Dict[str, Any]],
    processed_keys: Set[str],
    output_path: Path,
    usage_acc: UsageAccumulator,
    progress: ProgressBar,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    results = existing_results
    results_lock = threading.Lock()
    unsaved_count = 0

    def _on_done(record: Optional[Dict[str, Any]], task: Dict[str, Any]) -> None:
        nonlocal unsaved_count
        progress.update()
        if record is None:
            if verbose:
                doi_or_path = task["doi"] or task["article_path"]
                # silent skip for non-relevant; only show errors if needed
            return

        with results_lock:
            # Double-check dedup (another thread may have added same DOI)
            if task["identifier"] in processed_keys:
                return
            processed_keys.add(task["identifier"])
            results.append(record)
            unsaved_count += 1
            if verbose:
                cat_type = record.get('catalysis_type', '')
                if cat_type == 'heterogeneous':
                    detail = record.get('support') or 'none'
                else:
                    detail = record.get('ligand_name') or 'none'
                print(f"\n[+] {record['doi'] or record['source_path']} | "
                      f"{cat_type} metal={record['metal']} "
                      f"{'support' if cat_type == 'heterogeneous' else 'ligand'}={detail} "
                      f"type={record['coupling_type']}")
            if unsaved_count >= PERSIST_EVERY:
                write_results(results, output_path)
                unsaved_count = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_task = {
            pool.submit(_process_one, task, extractor, usage_acc): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                record = future.result()
            except Exception as exc:
                if verbose:
                    print(f"\nWorker error for {task['doi'] or task['article_path']}: {exc}")
                record = None
            _on_done(record, task)

    # Final flush
    with results_lock:
        write_results(results, output_path)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract coupling reaction info from article JSONs (multi-threaded).")
    parser.add_argument("--input-root", default="article_data", help="Root directory of article JSONs.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSON output path.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV_OUTPUT), help="CSV output path.")
    parser.add_argument("--limit", type=int, default=None, help="Max articles to process.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Number of parallel threads (default {DEFAULT_WORKERS}).")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output.")
    parser.add_argument("--model", default=SUPPORTED_MODEL, help="DeepSeek model name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    client = DeepSeekClient(model=args.model)
    extractor = CouplingInfoExtractor(client)
    output_path = Path(args.output).resolve()
    csv_path = Path(args.csv).resolve()

    existing_results, processed_keys = load_existing_results(output_path)
    initial_count = len(existing_results)
    if not args.quiet:
        print(f"Loaded {initial_count} existing records from {output_path}")

    # Scan & pre-filter articles
    article_paths = list(iter_article_files(input_root))
    if not args.quiet:
        print(f"Found {len(article_paths)} article files, filtering...")
    tasks = prepare_tasks(article_paths, processed_keys, limit=args.limit, verbose=not args.quiet)
    if not args.quiet:
        print(f"{len(tasks)} articles to process with {args.workers} workers")

    progress = ProgressBar(total=len(tasks), quiet=args.quiet)
    usage_acc = UsageAccumulator()

    try:
        results = analyze_articles(
            tasks,
            extractor,
            existing_results,
            processed_keys,
            output_path,
            usage_acc,
            progress,
            workers=args.workers,
            verbose=not args.quiet,
        )
    finally:
        progress.finish()

    # Final write
    write_results(results, output_path)
    export_csv(results, csv_path)

    if not args.quiet:
        new_count = len(results) - initial_count
        usage = usage_acc.as_dict()
        print(f"\nDone. Added {new_count} records (total {len(results)}).")
        print(f"JSON: {output_path}")
        print(f"CSV:  {csv_path}")
        print(f"Tokens — prompt: {usage['prompt_tokens']}, completion: {usage['completion_tokens']}, total: {usage['total_tokens']}")


if __name__ == "__main__":
    main()
