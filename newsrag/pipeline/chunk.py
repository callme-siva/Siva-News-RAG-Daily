"""Split a processed item into retrievable chunks (FR13).

Every chunk starts with a header (headline, region, category, source, date, URL) so it can
be cited and dated on its own. Bodies are split on sentence boundaries into pieces of about
`chunk_size` characters with `chunk_overlap` characters carried over.
"""

from __future__ import annotations

import re

from newsrag.ids import chunk_id
from newsrag.models import Chunk, Item, Processed

_SENTENCE_OR_NEWLINE = re.compile(r"[^.!?\n]+[.!?]*\s*|\n")


def header(item: Item, processed: Processed) -> str:
    lines = [
        f"Headline: {item.title}",
        f"Region: {item.region} | Category: {processed.category} | Source: {item.source} | "
        f"Published: {item.published_at.date().isoformat()} | URL: {item.url}",
    ]
    if item.also_reported_by:
        lines.append(f"Also reported by: {', '.join(item.also_reported_by)}")
    names = processed.entities.flat()
    if names:
        lines.append(f"Entities: {', '.join(names[:12])}")
    return "\n".join(lines)


def _pieces(text: str, size: int, overlap: int) -> list[str]:
    parts = _SENTENCE_OR_NEWLINE.findall(text) or [text]
    pieces: list[str] = []
    current = ""
    for part in parts:
        if current.strip() and len(current) + len(part) > size:
            pieces.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            current = (tail[tail.find(" ") + 1 :] if " " in tail else tail) + part
        else:
            current += part
    if current.strip():
        pieces.append(current.strip())
    return pieces


def build_chunks(item: Item, processed: Processed, size: int, overlap: int) -> list[Chunk]:
    """Chunk IDs use `processed.item_id`, the stored identity. It equals `item.item_id` at
    ingest, but stays correct on reindex even if URL-normalisation rules changed since."""
    head = header(item, processed)
    body_parts = [f"Summary: {processed.summary}"]
    if processed.key_facts:
        body_parts.append("Key facts: " + "; ".join(processed.key_facts))
    if item.body and item.body not in processed.summary:
        body_parts.append(item.body)
    pieces = _pieces("\n".join(body_parts), size, overlap)
    meta = {
        "region": item.region,
        "category": processed.category,
        "source": item.source,
        "published_ts": int(item.published_at.timestamp()),
    }
    return [
        Chunk(
            chunk_id=chunk_id(processed.item_id, i),
            item_id=processed.item_id,
            chunk_index=i,
            text=f"{head}\n[Part {i + 1} of {len(pieces)}]\n{piece}",
            metadata=meta,
        )
        for i, piece in enumerate(pieces)
    ]
