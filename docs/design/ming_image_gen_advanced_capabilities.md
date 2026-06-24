# Ming Image Generation Advanced Capability Audit

This note captures the pre-PR decision for the advanced image-generation files
that were moved out of Phase 1. The default production path stays
thinker-fused. Restored advanced paths must be explicit and observable.

| File | Role | PR decision | Why |
| --- | --- | --- | --- |
| `semantic_encoder.py` | Standalone semantic conditioning | Restore now as an explicit reference/debug path | Needed to compare thinker-fused condition embeddings against the independent BailingMoe path. |
| `bailing_moe_model.py` | Standalone BailingMoe implementation | Restore now, private to standalone encoder | Not a fallback; required by `MingSemanticEncoder`. |
| `bailing_moe_config.py` | Standalone BailingMoe config | Restore now, private to standalone encoder | Required to instantiate the standalone encoder. |
| `byt5_encoder.py` | Text-rendering enhancement | Restore behind an explicit flag | Useful for quoted/rendered text, but must not affect the default path. |
| `sd3_backend.py` | Alternate backend | Restore as an explicit experimental backend | Not Ming's main path; useful for backend interface validation. |

Default behavior:

- `semantic_source="thinker"` uses captured serving-thinker hidden states.
- `semantic_source="standalone"` is allowed only when standalone semantic
  encoding is explicitly enabled.
- ByT5 text rendering runs only when explicitly enabled.
- Missing semantic conditioning is an error, not a text-only/random fallback.
