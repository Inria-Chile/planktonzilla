---
name: inria-ui
description: "Apply Inria's official visual identity (charte graphique) when designing, building, or restyling ANY UI for an Inria product — web app, dashboard, Hugging Face Space, data-viz, slide, or doc. Covers the Palette 2024, Inria Sans/Serif typography, the République Française State lockup, logo rules, the graphic system (blanc tournant, dot-grid motif, gradients), components, and accessibility. Use whenever building UI for Inria / Project OcéanIA / planktonzilla, or when the user mentions Inria branding, the charte graphique, the Inria palette, Inria Sans/Serif, or the Inria red."
trigger: /inria-ui
---

# /inria-ui — build UI in the Inria visual identity

## When to use

Any time you design, build, or restyle a user interface for an **Inria** product — a web app,
dashboard, Hugging Face Space, data-visualisation, slide, or document — or when the user mentions
Inria branding, the *charte graphique*, the Inria palette, Inria Sans/Serif, or the Inria red.

## How to use

1. **Fill the INPUTS block** below (product name, emitter mode, output medium, …).
2. **Apply the ESSENTIALS** — they are sufficient for most screens on their own.
3. **For the complete contract** (full type scale, DSFR co-signature construction, gradient/scrim
   rules, complete light+dark token set, extended data-viz palette, full do/don't) read
   **`docs/INRIA_UI_PROMPT.md`** in this repo, and paste it as guidance for anything non-trivial.
4. Ship **light + dark**, **WCAG AA**, self-contained and token-driven; **state any assumption** you make.

---

## ESSENTIALS (self-contained)

### INPUTS — set these first
```
PRODUCT_NAME  : <e.g. "Plankton Atlas">
DOMAIN_ACCENT : <secondary brand hue; default Bleu mat>
EMITTER_MODE  : <sole | co-branded>    # sole → the République Française State lockup is MANDATORY
OUTPUT_MEDIUM : <web | csp-artifact | slide | doc>   # csp-artifact = no external hosts allowed
LOGO_ASSET    : <inline SVG provided? path? or "text facsimile" if none>
THEME         : <light | dark | both>  # default both
```

### Principles
Restraint + **one confident signal**: Rouge Inria is a *signal, not a surface* — one red element
per view (brand mark > primary action > selected > corner marker). Editorial **Inria Serif** titles
+ functional **Inria Sans** UI. Accessibility (AA) and honest data outrank decoration.

### Colour — Palette 2024 (immutable; identical in light & dark)
| Rouge Inria `#C9191E` · Framboise `#A60F79` · Violet `#534B9A` · Bleu mat `#27348B` · Bleu canard `#1067A3` |
- **The one red is `#C9191E`.** **Reject** `#E53516` / `#FF5636` / any "brighter" red.
- **White text on a brand fill → OK; black text on a brand colour → NEVER.** Brand colours also work
  as text on white. On dark, use brand hues as fills (white text) or their tints for text/accents.
- Lighter needs → sanctioned **70 / 50 / 20 % tints** via `color-mix(in srgb, var(--hue) 70%, var(--panel))`.

### Typography
- **Inria Serif** (Georgia fallback) for display/editorial titles; **Inria Sans** (Tahoma fallback)
  for all UI/body/data. Signature heading = bold Inria Sans + Inria Serif *italic* tail (Violet/Bleu mat).
- `--serif:"Inria Serif",Georgia,serif;  --sans:"Inria Sans",Tahoma,ui-sans-serif,system-ui,sans-serif;`
- `csp-artifact` → embed woff2 as `@font-face` `data:` URIs. Note: Google Fonts ships Inria Sans only
  300/400/700 (no ExtraBold).

### Logo & République Française
- Inria logo = **red script wordmark `#C9191E`** (white on dark); never render in ink/black, redraw,
  recolour, or distort — use the official SVG (or a flagged Inria-Serif-at-Rouge text facsimile if none).
- **When `EMITTER_MODE = sole`, the "RÉPUBLIQUE FRANÇAISE" State bloc-marque is MANDATORY** (Inria is
  a State operator): tricolour + "RÉPUBLIQUE FRANÇAISE" + "Liberté · Égalité · Fraternité", top-left,
  co-signed with Inria. Its DSFR colours are **State exceptions outside the palette**: blue `#000091`,
  red `#E1000F`, white `#FFFFFF`. Reproduce it, don't restyle.

### Graphic system
- **Blanc tournant = white** margin around content (≈24/48/64px); never bleed to the edge; max width ≈1200px.
- Signature **dot-grid motif** (include at least this), CSP-safe:
  ```css
  .inria-motif{ background-image:radial-gradient(currentColor 1px, transparent 1.5px);
    background-size:16px 16px; color:var(--bleu-mat); opacity:.08; } /* .11 on dark; corner tile, never behind text */
  ```
- Optional: diagonal-cropped **visuel** (one site-wide ~6–8° angle + text scrim), **dégradé** gradient at
  135° in brand hues (never Rouge), **red corner marker** bottom-right.

### Components
- **Primary button** = solid Rouge + white text (one per view). Secondary = neutral outline. Ghost = Rouge text.
- **Uniform focus:** 2px solid **Bleu canard** ring, 2px offset, on every interactive element.
- **Danger** = tinted panel + red border + **icon + label** (never a solid brand-red fill — it must not read as the primary action).
- Cards = white surface, hairline border, `--r-md`, soft shadow, optional corner marker. Icons: Lucide/Tabler, currentColor.

### Data-viz
- **Fixed order, Rouge reserved:** Bleu mat → Bleu canard → Violet → Framboise; use Rouge only for the
  highlighted/selected series. Extend with 70/50 % tints before muted grey `#AAB3BF` ("other"). Magnitude →
  single-hue tint ramp; polarity → two hues + grey midpoint; **never rainbow**. Always a **non-colour cue**.

### Token block
```css
:root{
  --rouge:#c9191e; --framboise:#a60f79; --violet:#534b9a; --bleu-mat:#27348b; --bleu-canard:#1067a3;
  --data-other:#aab3bf;
  --page:#ffffff; --panel:#ffffff; --sunken:#f4f6f8;
  --ink:#171a1d; --ink-2:#3f474e; --ink-muted:#5c666f; --hair:#e0e5ea; --border-strong:#b9c1ca;
  --ok:#2f8f66; --warn:#c77f2e; --danger:#a5231a; --info:#27348b;
  --serif:"Inria Serif",Georgia,serif; --sans:"Inria Sans",Tahoma,ui-sans-serif,system-ui,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,monospace;
  --r-sm:6px; --r-md:10px; --r-lg:16px; --r-pill:999px;
}
/* dark: keep brand hexes UNCHANGED; remap neutrals only; also emit :root[data-theme="dark"] for manual toggles */
@media (prefers-color-scheme:dark){:root{
  --page:#121517; --panel:#1a1e21; --sunken:#0e1113; --ink:#e9edf0; --ink-2:#b7c0c8; --ink-muted:#8a939c;
  --hair:#2a3034; --border-strong:#414b52; --ok:#48b184; --warn:#d59a4f; --danger:#e0645a; --info:#5f8fc4;
}}
```

---

**For the complete contract, read [`docs/INRIA_UI_PROMPT.md`](../../../docs/INRIA_UI_PROMPT.md).**

*Source: Inria charte graphique §4.13.1 (Logo) + §4.13.2 (Univers graphique — Palette 2024).
Inria Sans/Serif by Black[Foundry], SIL OFL. RF/DSFR colours per the Système de design de l'État.
Confirm exact logo clear-space, minimum sizes, and print CMYK against the current charter before production.*
