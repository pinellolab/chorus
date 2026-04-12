"""Analysis request metadata — preserves the user's question on every report.

The goal is that a user (or another teammate) who opens a report a month
later can immediately see:

    - What question the user actually asked (original natural-language prompt)
    - Which oracle + normalizer produced the numbers
    - Which tracks / cell types were considered
    - When the report was generated
    - Any caveats the tool flagged (missing data, fallbacks, etc.)

All fields are optional so existing callers keep working. MCP tool wrappers
construct an :class:`AnalysisRequest` from the user's prompt and pass it
through to the report builders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class AnalysisRequest:
    """Metadata describing the user's original question and how it was run.

    Attaching this to every report means a user re-opening an HTML/MD file
    later can tell exactly what was asked, what tool was used, and what
    limitations applied — without hunting through logs.
    """

    user_prompt: str | None = None
    tool_name: str | None = None            # e.g. "analyze_variant_multilayer"
    oracle_name: str | None = None
    normalizer_name: str | None = None      # e.g. "alphagenome pertrack", "none"
    tracks_requested: str | None = None     # human description: "all tracks", "K562 DNASE/CAGE"
    cell_types: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # caveats / limitations
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    )

    # -- serialisation --------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "user_prompt": self.user_prompt,
            "tool_name": self.tool_name,
            "oracle_name": self.oracle_name,
            "normalizer_name": self.normalizer_name,
            "tracks_requested": self.tracks_requested,
            "cell_types": list(self.cell_types),
            "notes": list(self.notes),
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "AnalysisRequest | None":
        if not data:
            return None
        return cls(
            user_prompt=data.get("user_prompt"),
            tool_name=data.get("tool_name"),
            oracle_name=data.get("oracle_name"),
            normalizer_name=data.get("normalizer_name"),
            tracks_requested=data.get("tracks_requested"),
            cell_types=list(data.get("cell_types") or []),
            notes=list(data.get("notes") or []),
            generated_at=data.get("generated_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        )

    # -- rendering ------------------------------------------------------

    def to_markdown(self) -> str:
        """Render as a markdown block suitable for the top of a report."""
        lines: list[str] = ["## Analysis Request", ""]
        if self.user_prompt:
            # Quote-block so multi-line prompts render correctly
            for row in self.user_prompt.strip().splitlines():
                lines.append(f"> {row}")
            lines.append("")
        rows: list[tuple[str, str]] = []
        if self.tool_name:
            rows.append(("Tool", f"`{self.tool_name}`"))
        if self.oracle_name:
            rows.append(("Oracle", self.oracle_name))
        if self.normalizer_name:
            rows.append(("Normalizer", self.normalizer_name))
        if self.tracks_requested:
            rows.append(("Tracks requested", self.tracks_requested))
        if self.cell_types:
            rows.append(("Cell types", ", ".join(self.cell_types[:8]) + (" …" if len(self.cell_types) > 8 else "")))
        rows.append(("Generated", self.generated_at))
        for label, value in rows:
            lines.append(f"- **{label}**: {value}")
        if self.notes:
            lines.append("")
            lines.append("**Notes / caveats:**")
            for note in self.notes:
                lines.append(f"- {note}")
        lines.append("")
        return "\n".join(lines)

    def to_html_fragment(self) -> str:
        """Render as an HTML block suitable for embedding at the top of a report."""
        import html as _html

        parts: list[str] = [
            '<section class="analysis-request" style="background:#f6f8fa;'
            'border-left:4px solid #0366d6;padding:12px 16px;'
            'margin:0 0 16px 0;border-radius:4px;font-size:0.92em;">',
            '<h3 style="margin:0 0 8px 0;font-size:1.05em;">Analysis Request</h3>',
        ]
        if self.user_prompt:
            safe = _html.escape(self.user_prompt.strip()).replace("\n", "<br>")
            parts.append(
                f'<blockquote style="margin:0 0 8px 0;padding:6px 10px;'
                f'background:#fff;border-left:3px solid #d1d5da;'
                f'color:#24292e;">{safe}</blockquote>'
            )
        parts.append('<ul style="margin:0;padding-left:20px;list-style:disc;">')

        def _row(label: str, value: str) -> str:
            return (
                f'<li><strong>{_html.escape(label)}:</strong> '
                f'{_html.escape(value)}</li>'
            )

        if self.tool_name:
            parts.append(_row("Tool", self.tool_name))
        if self.oracle_name:
            parts.append(_row("Oracle", self.oracle_name))
        if self.normalizer_name:
            parts.append(_row("Normalizer", self.normalizer_name))
        if self.tracks_requested:
            parts.append(_row("Tracks requested", self.tracks_requested))
        if self.cell_types:
            ct_str = ", ".join(self.cell_types[:8]) + (" …" if len(self.cell_types) > 8 else "")
            parts.append(_row("Cell types", ct_str))
        parts.append(_row("Generated", self.generated_at))
        parts.append("</ul>")
        if self.notes:
            parts.append(
                '<p style="margin:8px 0 0 0;"><strong>Notes / caveats:</strong></p>'
            )
            parts.append('<ul style="margin:4px 0 0 0;padding-left:20px;">')
            for note in self.notes:
                parts.append(f'<li>{_html.escape(note)}</li>')
            parts.append("</ul>")
        parts.append("</section>")
        return "\n".join(parts)
