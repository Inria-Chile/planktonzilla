<!--
  Canonical Inria UI design prompt for Project OcéanIA / planktonzilla.
  Derived from the Inria charte graphique — §4.13.1 (Logo : utilisation et formats) and
  §4.13.2 (Univers graphique: Palette 2024, Typographie, Trames/motifs, Système graphique) —
  and the project palette (planktonzilla/explorer/sankey.py, planktonzilla/app.py).
  Mirrored by the /inria-ui skill at .github/skills/inria-ui/SKILL.md.
-->

# Inria Visual Identity — Reusable UI Design Prompt

> Paste this as system/context guidance whenever you ask Claude to design or build UI for an
> **Inria** product (web app, dashboard, Hugging Face Space, data-viz, slide, doc). It encodes
> Inria's official *charte graphique* — §4.13.1 *Logo : utilisation et formats* and §4.13.2
> *Univers graphique* (Palette 2024, Typographie, Trames/motifs, Système graphique). Follow it
> exactly. **Fill in the INPUTS block first**, then apply every rule. Where a value is still
> unspecified, choose the option most consistent with these principles and say so.

---

## INPUTS — set these first (they change the output materially)

```
PRODUCT_NAME     : <the product/app name, e.g. "Plankton Atlas">
DOMAIN_ACCENT    : <which of the 5 brand hues is this product's secondary accent; default Bleu mat>
EMITTER_MODE     : <sole | co-branded>   # sole → the République Française State lockup is MANDATORY
CO_EMITTERS      : <if co-branded: partner logos/order, e.g. "Sorbonne Université, CNRS">
OUTPUT_MEDIUM    : <web | csp-artifact | slide | doc>   # csp-artifact = no external hosts allowed
LOGO_ASSET       : <inline SVG / path; else the official Inria logo https://inria.fr/themes/custom/inria/logo/logo.svg; else "text facsimile">
THEME            : <light | dark | both>   # default both
```

Reference these tokens throughout — do **not** bake one product's specifics into the rules.

## 0 · Principles (the feel)

1. **Restraint, then one confident signal.** The identity is clean, neutral, carried by generous
   whitespace and precise type. **Rouge Inria is a signal, not a surface** — it marks the single
   most important action, the brand mark, the active/selected state, or the optional corner marker.
   Never flood a large area or page background with it, and never let more than one element read as
   "the red" per view (precedence: brand mark > primary action > selected state > corner marker;
   demote lower ones to neutral/tint when they'd collide).
2. **Editorial serif + functional sans.** Inria Serif gives titles a considered, scholarly voice;
   Inria Sans runs everything you operate. The pairing *is* the identity.
3. **Scientific clarity & accessibility.** These are research instruments. Legibility, honest data
   encodings, and **WCAG AA** (4.5:1 text / 3:1 UI) outrank decoration. The charter rates every
   brand colour for text legibility — respect those ratings and **compute exact contrast at build
   time** rather than trusting any table.

## 1 · Colour — Inria Palette 2024 (authoritative, immutable)

The five brand colours are **fixed** and identical in light **and** dark themes — never recolour
them. Only *neutrals* change between themes. Values below (RGB is the source of truth; hex derived).

| Name | HEX | RGB | CMYK (print) |
|------|-----|-----|--------------|
| **Rouge Inria** | `#C9191E` | 201, 25, 30 | M100 J100 |
| **Framboise** | `#A60F79` | 166, 15, 121 | — |
| **Violet** | `#534B9A` | 83, 75, 154 | — |
| **Bleu mat** | `#27348B` | 39, 52, 139 | — |
| **Bleu canard** | `#1067A3` | 16, 103, 163 | — |

- **The one red is `#C9191E`.** Reject `#E53516` / `#FF5636` and any other "brighter" red — they
  are off-charter and fail white-text contrast. If you need a lighter red, use a **tint** (below),
  never a new hue.
- **Tints** — each hue has sanctioned **70% / 50% / 20%** steps. Define them as an opaque mix with
  the current surface so they stay predictable and AA-testable:
  `--{hue}-70: color-mix(in srgb, var(--{hue}) 70%, var(--panel));` (and 50%, 20%). Tints are for
  large fills, backgrounds, and chart ramps; keep full-strength colour for accents.
- **Contrast rules (obey; verify exact ratios at build):**
  - **White text on any full-strength brand colour → passes AA.** Always pair brand fills with
    **white** text.
  - **Brand colour as text on white → passes** (all five ≥ ~5:1; Bleu mat highest ~10:1, Rouge
    lowest ~5–6:1). Do not set brand-colour text on a *tinted* or coloured ground without checking.
  - **Black text on a brand colour → FAILS. Never.**
  - On **dark** surfaces, use brand hues as **fills with white text**, or use their **lighter tints**
    for text/accents — full-strength brand text on a dark panel is too low-contrast.
- **CMYK/print:** hex/RGB are for screen; use CMYK for edition/.ai/.eps assets (Rouge = M100 J100).

## 2 · Neutrals & tokens (UI scaffolding)

The palette poster is colour-only; UI needs neutrals. Keep them cool-neutral and clean — the
charter leans on **white** (see *blanc tournant*).

- **Blanc tournant = white.** The page/outer margin surrounding content is **`#FFFFFF`** in light
  mode. Differentiate cards/panels with **hairlines and soft shadows**, not a grey canvas.
- Provide the full token set for **both** themes (a partial dark override is a bug — secondary text
  and semantics must be re-tuned):

```css
:root{
  /* brand (immutable in both themes) */
  --rouge:#c9191e; --framboise:#a60f79; --violet:#534b9a; --bleu-mat:#27348b; --bleu-canard:#1067a3;
  --data-other:#aab3bf;                       /* reserved "other/unknown" */
  /* neutrals — light */
  --page:#ffffff; --panel:#ffffff; --sunken:#f4f6f8;
  --ink:#171a1d; --ink-2:#3f474e; --ink-muted:#5c666f;   /* ink-muted passes AA on white */
  --hair:#e0e5ea;                              /* decorative dividers only */
  --border-strong:#b9c1ca;                     /* input/control edges — meets 3:1 */
  /* semantic (distinct from brand; always with icon+label) */
  --ok:#2f8f66; --warn:#c77f2e; --danger:#a5231a; --info:#27348b;
  /* elevation */
  --shadow-1:0 1px 2px rgba(23,26,29,.06),0 1px 3px rgba(23,26,29,.10);
  --shadow-2:0 4px 12px rgba(23,26,29,.10),0 2px 4px rgba(23,26,29,.06);
  /* type */
  --serif:"Inria Serif",Georgia,serif;
  --sans:"Inria Sans",Tahoma,ui-sans-serif,system-ui,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,monospace;
  /* shape */
  --r-sm:6px; --r-md:10px; --r-lg:16px; --r-pill:999px;
}
/* dark — brand hexes UNCHANGED; only neutrals + shadow re-tuned. Emit both signals so a
   manual theme toggle (HF Space / Artifact stamps data-theme) wins over the OS query. */
@media (prefers-color-scheme:dark){:root{
  --page:#121517; --panel:#1a1e21; --sunken:#0e1113;
  --ink:#e9edf0; --ink-2:#b7c0c8; --ink-muted:#8a939c;
  --hair:#2a3034; --border-strong:#414b52;
  --ok:#48b184; --warn:#d59a4f; --danger:#e0645a; --info:#5f8fc4;
  --shadow-1:none; --shadow-2:0 8px 30px rgba(0,0,0,.45);   /* lean on borders, not shadows */
}}
:root[data-theme="dark"]{/* repeat the dark values here so toggles work */}
:root[data-theme="light"]{/* repeat the light values here */}
```

## 3 · Typography

- **Inria Serif** — display & editorial titles, hero lines, pull quotes; **italic** for emphasis.
- **Inria Sans** — all UI, functional headings, body, data, labels, controls.
- **Signature heading device:** bold **Inria Sans** lead + **Inria Serif italic** tail, often in
  Violet or Bleu mat — e.g. **“Lorem ipsum”** *dolor sit amet* (charter pattern).
- **Type scale** (rem; state family per level):

| Level | Size / line-height | Family |
|-------|--------------------|--------|
| Display | 3rem / 1.05 | Inria Serif 700 |
| H1 | 2.25rem / 1.12 | Inria Serif 700 |
| H2 | 1.75rem / 1.2 | Inria Sans 700 |
| H3 | 1.375rem / 1.25 | Inria Sans 700 |
| Body | 1rem / 1.55 | Inria Sans 400 |
| Small | 0.875rem / 1.45 | Inria Sans 400 |
| Label/overline | 0.6875rem / 1.4, +0.12em, uppercase | Inria Sans 600 |
| Data/mono | 0.8125rem | mono |

  Keep running text ≤ ~70ch; `text-wrap:balance` on headings.
- **Weights:** Light / Regular / Bold / ExtraBold (+ italics) exist in the retail family. **Note:**
  Google Fonts ships Inria Sans only at **300/400/700** — if you need ExtraBold (800), supply the
  face yourself or downgrade "ExtraBold" to **Bold**.
- **Fallbacks (official):** **Georgia** (serif) and **Tahoma** (sans) — the charter's designated
  office substitutes (ship on macOS/Windows). Add `size-adjust`/`ascent-override` to the fallback
  faces to avoid layout shift.
- **Font loading by `OUTPUT_MEDIUM`:**
  - `web` — prefer **self-hosting** Inria Sans/Serif (SIL OFL) for a French public-sector /
    RGAA-conscious site; Google Fonts linking is acceptable for internal tools.
  - `csp-artifact` — external font hosts are blocked; **embed** the `woff2` faces as `@font-face`
    **`data:` URIs** (the OcéanIA design-system artifact already ships an embeddable block).
  - `slide` / `doc` — install the OTFs or fall back to Georgia/Tahoma.
- **Monospace:** neutral system mono for numbers/IDs/code — the charter has no brand mono.

## 4 · Logo & the République Française State brand

- **Inria logo** is a **red script “Inria” wordmark** (Rouge `#C9191E`), reversed **white** on dark.
  **Never** render it in ink/black, redraw, recolour, distort, rotate, or add effects.
  - **Asset:** use the official pack — web (**.svg**, .png), office (.png/.jpg), edition (.ai/.eps).
    The canonical web SVG is <https://inria.fr/themes/custom/inria/logo/logo.svg> (embed it as a
    `data:` URI for CSP-restricted artifacts, where external hosts are blocked).
    If `LOGO_ASSET` provides an SVG/data-URI, embed it. If none is available (e.g. a CSP artifact
    with nothing supplied), a **text facsimile is permitted only** if set in **Inria Serif at Rouge
    `#C9191E`** (white on dark) and clearly flagged *non-production — replace with official SVG*.
  - **Clear space** ≈ the wordmark's cap-/x-height on all four sides; **minimum height ≈ 24px** on
    screen. (Confirm exact figures against the charter for print.)
- **République Française (Marque de l'État) — mandatory when `EMITTER_MODE = sole`.** Inria is a
  State operator, so its sole-emitter supports **must** carry the State bloc-marque: the tricolour
  fragment + **“RÉPUBLIQUE FRANÇAISE”** (bold) + the devise **“Liberté · Égalité · Fraternité.”**
  - It is a **controlled DSFR asset — reproduce it, don't restyle it.** Its colours are **State
    exceptions that sit OUTSIDE the Inria 5-colour palette:** blue `#000091`, red `#E1000F`, white
    `#FFFFFF`.
  - **Co-signature construction (emitter lockup, top-left):** RF bloc on the **left**, a thin
    vertical rule, then the **Inria** wordmark on the right, cap-heights matched; give the whole
    lockup its own clear space. Place it inside the *blanc tournant*, top-left; keep the product
    title beside/below it, never merged into the mark.
  - When `EMITTER_MODE = co-branded`, follow the partner co-branding lockup instead of forcing the
    RF block; the RF marque is required only when Inria is the sole emitter.

## 5 · Système graphique (the Inria/State layout framework)

- **Blanc tournant** — a **white** margin/frame around the content zone. Outer margin ≈ **24px
  mobile / 48px tablet / 64px desktop**; max content width ≈ **1200px**; never bleed content to the
  viewport edge.
- **Emitter lockup** top-left inside the blanc tournant (see §4). Header min-height ~64px desktop /
  56px mobile; below ~480px, stack the product title under the lockup — **never shrink the RF marque
  below its minimum or drop it.**
- Inside the content zone, Inria's recognisable devices. **Include at least the dot-grid motif** so
  the page reads as Inria (the other two are optional):
  1. **Motif (trame) — the signature dot-grid.** Provide it inline (CSP-safe):
     ```css
     .inria-motif{
       background-image:radial-gradient(currentColor 1px, transparent 1.5px);
       background-size:16px 16px;            /* dot Ø ~2px, pitch 16px */
       color:var(--bleu-mat);                /* nearest brand hue */
       opacity:.08;                          /* light; .11 on dark */
     }
     ```
     Use it as a ~240×240px **corner tile** in heroes, empty states, or section dividers — **never**
     tiled behind body text. (Charter also offers arrows/crosses/concentric-wave variants.)
  2. **Visuel** — a photographic block with a **consistent diagonal crop**. Fix ONE site-wide angle
     and orientation, e.g. `clip-path:polygon(0 0,100% 0,100% 92%,0 100%)` (~6–8° edge), reused
     everywhere. Over photos, add a **scrim** (`linear-gradient(0deg,rgba(23,26,29,.55),transparent)`
     or a Bleu-mat tint) and verify hero text ≥ 4.5:1 over the darkest region it covers.
  3. **Dégradé** — a **gradient** in brand hues at **135°** (echoing the diagonal). Sanctioned pairs:
     Violet→Bleu mat, Bleu mat→Bleu canard, or single-hue 100→70% tint. **Never use Rouge in a
     gradient** (keep red as an accent); require white heading text ≥ 4.5:1 across both stops.
- **Marqueur Inria (optional):** a small **Rouge `#C9191E` corner marker** (L-shaped block) at the
  **bottom-right** of a frame/card — a sparing brand flourish, counts as the view's "one red" if the
  primary action is styled neutrally.

## 6 · Components

- **Buttons** — *primary* = solid **Rouge Inria** + **white** text (one per view); *secondary* =
  neutral outline (`--border-strong`); *ghost/link* = Rouge text on transparent. **Focus (uniform):
  2px solid Bleu canard ring, 2px offset, ≥3:1** against adjacent surfaces — the *same* ring on
  every interactive element.
- **Inputs** — `--border-strong` edge; on focus, Rouge border **plus** the Bleu-canard ring.
- **Tags / chips** — brand-hue fill + white text, `--r-pill`, assigned in fixed data order.
- **Alerts / status** — semantic colours (`--ok/--warn/--danger/--info`), **always** with an icon
  **and** a text label, never colour alone. Because a red "danger" sits near Rouge Inria, style
  danger as a **tinted panel + red border + icon** (not a solid brand-red fill) so it never reads as
  the primary action.
- **Cards/panels** — white surface, hairline border, `--r-md`, `--shadow-1`, optional corner marker.
- **Icons** — one set (e.g. **Lucide** or **Tabler**), stroke 1.5–2px, sizes 16/20/24 on the 4px
  grid, colour via `currentColor` inheriting ink tokens.

## 7 · Space, shape, motion, layout

- **4px grid:** 4 · 8 · 12 · 16 · 24 · 32 · 48 · 64; lay out with flex/grid `gap`.
- **Radius:** sm 6 · md 10 · lg 16 · pill 999 (md default).
- **Breakpoints:** 480 / 768 / 1024 / 1280; **12-column grid, 24px gutters**; max content 1200px.
- **Motion:** 120–200ms `ease`; always honour `prefers-reduced-motion`.

## 8 · Data visualisation

- **Fixed categorical order — reserve Rouge for emphasis.** Because red is the brand "signal", do
  **not** lead a categorical ramp with it. Order: **Bleu mat → Bleu canard → Violet → Framboise**,
  and use **Rouge only for the highlighted/selected series or a KPI**. For a 5th+ base category add
  Rouge, then extend with the **70% / 50% tints** of the five hues (still on-brand, non-rainbow)
  before falling back to muted grey `#AAB3BF` for a true "other/unknown". Beyond ~10 series, prefer
  aggregation or small multiples.
- **Magnitude** → single-hue **tint ramp** (light→dark). **Polarity** → two brand hues + a
  neutral-grey midpoint. Never a rainbow.
- **Always a non-colour cue** (shape/pattern/label/direct label) — the palette is certified for
  white/black text contrast, *not* for colour-vs-colour discrimination, so colour alone is unsafe
  for colour-blind readers.
- Thin marks, recessive grid/axes, direct labels over legends where possible.

## 9 · Accessibility & output expectations

- **Contrast:** AA everywhere; **white-on-brand only**, never **black-on-brand**; body text in ink
  tokens, never brand-colour text on a coloured ground. Ink-muted (not Ink-3) for small secondary
  text; `--border-strong` (not hairline) for interactive edges.
- **Theming:** both light and dark from one token set; brand hexes immutable; manual theme toggle
  works (emit `data-theme` overrides as well as the media query).
- **When you build:** produce **self-contained, token-driven** code; define the palette as CSS
  variables; use the Inria Sans/Serif stacks with Georgia/Tahoma fallback and the right font-loading
  path for `OUTPUT_MEDIUM`; include the emitter lockup per `EMITTER_MODE`; keep motif/gradient
  restrained; ship light **and** dark; and **state any assumption you had to make.**

## 10 · Do / Don't

**Do:** fill the INPUTS block first · anchor each view on one Rouge-Inria signal · pair Inria Serif
display + Inria Sans UI · keep the five palette hexes fixed in both themes and use tints for lighter
needs · white text on brand fills · include the RF+Inria lockup when sole emitter · keep white
blanc-tournant margins · use the dot-grid motif subtly · assign data colours in fixed order (Rouge
reserved) with a non-colour cue · compute exact contrast at build.

**Don't:** flood surfaces/backgrounds with red · recolour the brand hues (no `#E53516`/`#FF5636`) ·
put black text on a brand colour · render the Inria wordmark in ink/black · omit the République
Française lockup when Inria is sole emitter · redraw/recolour/distort either mark · substitute
Arial/Inter/Roboto for the Inria family · lead a chart with red or rainbow-cycle a 6th hue · rely on
colour alone · bleed content to the edge (kills the blanc tournant) · leave a partial dark override
that hides secondary text.

---

*Source: Inria charte graphique — §4.13.1 Logo : utilisation et formats; §4.13.2 Univers graphique
(Palette 2024, Typographie, Trames/motifs, Système graphique). Inria Sans/Serif by Black[Foundry],
SIL OFL. República/DSFR State-mark colours per the Système de design de l'État. Confirm exact logo
clear-space, minimum sizes, and print CMYK against the current charter before production.*
