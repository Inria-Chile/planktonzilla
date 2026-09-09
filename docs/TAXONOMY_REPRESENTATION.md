# A more flexible and robust representation of the taxonomy mapping

*Research report, 2026-09-09. Subject: `planktonzilla/planktonzilla_dataset/planktonzilla_taxonomy.csv`, the
table that maps every source dataset's class directory onto the shared planktonzilla taxonomy. Nothing in this
document changes the CSV, the published dataset, or any code; it is the case for a change, the evidence behind it,
and a recommended design with a migration plan.*

Method, in one paragraph. The committed CSV and every code path that reads or writes it were audited from five
independent angles (data model, curation workflow, consumers and migration surface, scientific taxonomy, ML
labels and downstream artifacts), producing 53 findings. Four external surveys covered biodiversity data
standards, schema and validation tooling, how other ML datasets and the plankton-imaging community represent
label spaces, and ontology and graph tooling (49 candidates). The findings were distilled into a numbered
requirement list. Six representation designs were then written independently from fixed angles, each scored
by three judges with different lenses and attacked by an adversarial refuter; this document is the synthesis. Every number below was measured on the committed file (2,358 rows, sha256
`95de6c49…`) or reproduced from the code, and is cited as `path:line` where it comes from code.

---

## 1. What the CSV is today

One row per `(Dataset, Raw_Labels)` pair, i.e. per class directory of one source's imagefolder. Nineteen
columns, in this order:

```
Dataset, Raw_Labels,
Kingdom, Phylum, Class, Order, Family, Genus, Species,
proposed_label, plankton, living, root_class, qualifier,
wikidata_ID, aphia_ID, NCBI_ID, BOLD_ID, ecotaxa_ID
```

| Column group | Meaning | Measured on the committed file |
| --- | --- | --- |
| `Dataset` | source name = `datasets[].name` in the registry = `dataset` column of the published set | 21 sources; 15 in the published v1.0, 6 pending |
| `Raw_Labels` | the class-directory name, byte-exact | 1,627 distinct; 378 shared by more than one source; 1,842 with uppercase, 229 with commas (FREPJ), 268 with `<` (EcoTaxa), 378 with spaces |
| `Kingdom … Species` | seven fixed Linnaean rank slots, lowercase; `Species` holds the epithet only | 1,297 distinct rank-path nodes; depth used by living rows: 7→472, 6→599, 5→224, 4→238, 3→240, 2→160, 1→35, 0→45; zero rank gaps |
| `proposed_label` | the harmonised class name | 907 distinct; each has exactly one lineage; 319 are used by two or more sources and cover 1,735 rows (74 %) |
| `plankton`, `living`, `root_class`, `qualifier` | what kind of object the image shows and whether it counts as plankton | `root_class` ∈ living 2,013 / detritus 165 / artefact 158 / inert 22; `living` ⇔ `root_class == living` with zero exceptions; 14-term `qualifier` vocabulary |
| five `*_ID` columns | crosswalks to Wikidata, WoRMS, NCBI, BOLD and a legacy EcoTaxa id space | the three numeric ids are float-serialised in every non-empty cell (`135336.0`); `ecotaxa_ID` holds `;`-joined lists in 598 cells |

The single reader of record is `generate_planktonzilla.build_taxonomy_lookup`
(`planktonzilla/planktonzilla_dataset/generate_planktonzilla.py:122-172`). It returns
`{(Dataset, Raw_Labels): {16 columns}}` with blanks as `None`, numeric ids cast to decimal-free strings, and a
duplicate key warned about and resolved last-wins. `RedefineDataset._taxonomy_row` then LEFT-joins every one of
the 17.4 M published image rows onto it (`:305-334`); a key that is absent yields sixteen `None`s silently.
`update_planktonzilla.build_sync_dict` re-applies the same lookup to carried-over rows. Everything the published
dataset, the paper and the four released CLIP models know about taxonomy comes through that one function, and
the training label of the released models is `" ".join(non-empty ranks)` over the published columns
(`gen_planktonzilla_only_plankton.py:81-96`), which is also the CLIP caption (`save_planktonzilla_for_clip.py:107`).

Seven other modules parse the file on their own rather than through that reader: `frepj_validate.py:113-131`
(a near-verbatim copy), `sankey.py:575-579`, `utils/verify_taxonomy_ids.py:522-535`,
`utils/build_frepj_taxonomy.py:151-209`, `utils/build_tara_pacific_taxonomy.py:435-459`,
`utils/resolve_frepj_ids.py:116-119`, `utils/extract_taxon_ids.py:256-277`, and nine test files read it directly.
Three of those modules also *write* it, by locating their own block in the byte stream: one of them
(`build_frepj_taxonomy.write_csv`, `:434-454`) discards every block after its own on a no-op re-run and blanks
208 backfilled id cells, exiting 0 (reproduced; `docs/CODE_REVIEW.md` finding 1.1).

---

## 2. Why the wide CSV is neither flexible nor robust

The audit findings group into six structural problems. Each is a property of the *representation*, not a
curation mistake, which is why fixing rows one at a time cannot resolve them.

### 2.1 Facts about a taxon are stored once per source label, not once per taxon

`proposed_label` determines the seven rank columns with zero exceptions, yet the lineage is written out
2,358 times: 10,157 redundant rank cells. Re-parenting `bacillariophyceae` means editing 300 rows;
`copepoda` is spelled out on 40 rows across 15 sources. The three id-curation commits in the file's history
rewrote 229, 12 and 85 whole 19-field lines to change only id cells, and a reviewer cannot tell an id change from
a lineage change in the diff. Partial application of such edits is the documented origin of the one place where
the dependency *does* break: 10 labels (97 rows, all FREPJ) carry a different id tuple from the same label in
other sources, because ids were blanked per row instead of per taxon (commit 7262085). A normalised model makes
that state unrepresentable; the frozen artifact, however, has to keep reproducing it (§4).

### 2.2 Three different things share one row

The row fuses *which organism* (the taxon), *what the image shows* (a whole organism, a part, an egg, a
larva, something that merely looks like it, detritus, an artefact) and *a source-specific judgement* (the
`plankton` flag). Measured: 26 labels occur with more than one `root_class` (`annelida` is living on whole-body
rows and detritus on `part_Annelida`), 58 with more than one `qualifier`, and `plankton` is not even a
function of `(label, root_class, qualifier)`: adult fish are `False` in jedioceans and zoolake while fish larvae
elsewhere are `True`, plus the two KI-10 fish-egg contradictions. So `root_class`, `qualifier` and `plankton`
are properties of the mapping (or of a class concept), never of the taxon, and any representation that hangs
them on the taxon silently merges 484 rows into the wrong class. 86 non-living rows carry a full lineage: they
are parts of organisms and the lineage names the organism the part belongs to.

### 2.3 The name string is the identity

There is no identifier for a taxon other than its lowercase name. Consequences, all present in the file:

- **Homonyms resolve to the wrong organism.** Every comb-jelly class directory in 9 sources (12 rows) is
  published under the *diatom* genus *Ctenophora* (`chromista > heterokontophyta > bacillariophyceae >
  fragilariales`, aphia 163921), and the CLIP caption for those images reads accordingly. `LClass_siphonophora`
  maps to a millipede genus; `Cladocera` in zoocamnet to the bivalve genus *Cladoceramus* (6,669 images). The
  one-name-one-lineage invariant that keeps the file consistent is exactly what forbids a second *Ctenophora*
  node; `build_tara_pacific_taxonomy.py:340-356` propagates the diatom lineage knowingly for that reason.
- **Synonyms become two classes.** `Heterocapsa_triquetra` maps to *heterocapsa triquetra* in two sources and to
  its accepted name *kryptoperidinium triquetrum* in a third, with two different families; *Neoceratium* /
  *Tripos* likewise (104,887 images).
- **Rank-slot contamination cannot be prevented.** The same name sits in two or three rank slots of one row in 4
  rows (KI-8); the authority tooling finds 23 such (name, slot) combinations across 117 taxa.
- **The epithet-only `Species` column is not a node identity.** 19 epithets occur under more than one genus;
  `pz_sankey` has to re-prefix the genus (`sankey.py:128-141`).
- **The implied tree is not a tree.** Order `arcellinida` sits under two Classes; families `bosminidae`,
  `daphniidae` and `sididae` each under two Orders, because different sources' curators followed different
  classification systems (WoRMS superorder *Diplostraca* vs the suborders *Anomopoda* / *Ctenopoda*). The
  flat row cannot say which system it follows, so contradictions are indistinguishable from errors, and 1,335
  expected NCBI-vs-WoRMS differences have to be waived to keep the authority check green.

### 2.4 Seven fixed rank columns lose precision

Subphylum *Crustacea* (15 rows) collapses to *Arthropoda*; *Rhizaria*, *Retaria* and *SAR* collapse to
*Chromista* at Kingdom depth; `Eukaryota` has no slot at all and is stored with every rank blank (KI-9). Six taxa
whose own rank lies outside the seven are pushed into the nearest slot (infraorders *Brachyura* / *Achelata* as
Orders, tribe *Alciopini* as a Family, subclass *Cirripedia* as a Class), which is why 12 curated labels are not
the deepest node of their own lineage and 10 training-label strings silently absorb 12 finer labels
(`animalia arthropoda malacostraca decapoda` ← *decapoda*, *brachyura*, *achelata*). The training vocabulary
built from these columns has 849 classes where the curated table has 861.

### 2.5 The process, not the format, provides the safety

"Append-only, first 1,486 lines byte-frozen" is enforced by a sha256 over physical line offsets
(`tests/test_frepj_taxonomy_coverage.py:169-177`), by hard-coded row sums in three tests, and by each builder
re-deriving "where my block starts" from the byte stream. The first 1,485 rows are sorted by `proposed_label`
with 15 sources interleaved in 1,071 runs; the rows after them are six contiguous blocks appended at EOF, which
is why the last two source additions produced a hand-resolved merge conflict on the same hunk. Adding a source
has had three precedents with three different methods (a 937-line builder with rule tables in Python; a
builder plus a Markdown-embedded provenance table; 44 hand-typed lines whose only provenance is a test
docstring). Provenance of *why a row says what it says* exists for 229 of 2,358 rows, inside a Markdown file;
the Tara Pacific builder computes it for all 600 of its rows and writes out 95. The authority snapshot is
keyed to the sha256 of the whole CSV, so correcting a spelling that involves no id forces a network re-harvest
and an 86,618-line JSON diff in the same PR.

### 2.6 Downstream label vocabularies are an emergent side-effect

Class ids of the released models are alphabetical positions of `" ".join(ranks)`; landing the six pending
sources would move the integer id of 591 of the 598 existing classes with no crosswalk. The `Kingdom != ""` filter
admits `None`, so five `plankton=True` rows with no Kingdom (516 images in v1.0) form an empty-string class at
index 0. Life stage and morphology (87,031 images with a non-`full_body` qualifier) never reach the caption.
The Sankey, the training label and the caption are three separate code-defined derivations of the same seven
columns.

---

## 3. What has to survive: the frozen contract

`planktonzilla-17M` v1.0, the paper and the four released models are pinned to today's column values, defects
included, under the repository's zero-behavioural-drift rule (KNOWN_ISSUES KI-8..KI-13 are pinned as-is by
`tests/test_taxonomy_known_issues.py`). Any new representation is therefore judged first on whether it can
*regenerate* the old one exactly, and only then on what else it can do. The prototype in §5 shows this is
achievable. Concretely, the contract is:

- the 19-column CSV, byte-for-byte: LF line endings, `csv.QUOTE_MINIMAL`, the header order above, today's
  physical row order (the interleaved sorted prefix, then frepj 229 / daplankton 44 / tara_pacific 600), the
  `X.0` id form, the `;`-joined `ecotaxa_ID` lists in their original item order;
- the 16-column lookup of `build_taxonomy_lookup`, value- **and Python-type**-identical for all 2,358 keys
  (`tests/test_taxonomy_lookup_equivalence.py` already pins this against a frozen pandas implementation);
- the published projection in `constants.CONSOLIDATED_COLUMNS` order (note the id order there is wikidata,
  ecotaxa, aphia, NCBI, BOLD, unlike the CSV header), `plankton` as `bool`, `living` excluded;
- the label-string projection: 849 distinct `" ".join(non-empty ranks)` strings over the 21 sources, 598 over
  the 15 v1.0 sources, in `sorted(set())` order, including the six repeated-token strings caused by KI-8 and
  the empty-string class;
- the 13 rows whose id tuple diverges from their label's canonical tuple, the Harpacticoida / Creseidae /
  Cladocera mislabels, the *Ctenophora* homonym, the `Eukaryota` casing: all reproduced unchanged until a
  deliberate, versioned data release corrects them.

Legitimate conventions that must **not** be "fixed" by a representation change: byte-exact `Raw_Labels`
(uppercase, commas, `<`, spaces, misspellings); shared species epithets; tautonyms; Chromista/Myzozoa (WoRMS)
where NCBI has neither; FREPJ `_unk` sentinels collapsing to the parent rank; blank aphia ids for freshwater
taxa (WoRMS is marine); the legacy `ecotaxa_ID` space that no longer resolves against EcoTaxa (never backfill
it); the lookup's duplicate-key last-wins and missing-key all-`None` behaviour (guard at the store, not in the
mapper); test fixtures that fabricate 18-column CSVs with capitalised ranks and `root_class: zoo` (the loader
must keep accepting them).

---

## 4. Requirements

Seven MUST requirements (blockers for adoption), fifteen SHOULD requirements, three MAY. Each traces to the
audit lens that produced it (DM data model, CW curation workflow, CM consumers/migration, ST scientific
taxonomy, ML labels).

**MUST**

- **M1 Lossless legacy reproduction.** A tested exporter regenerates the 19-column CSV byte-for-byte and the
  16-column lookup value-and-type-identical, defects and the 13 id overrides included; the golden test runs in
  CI before any consumer switches and until the wide CSV is retired. [DM, CW, CM, ST, ML]
- **M2 Three-layer model.** Taxon/concept (identity independent of name; lineage stored once; ids stored once)
  ← class concept `(taxon-or-none, root_class, qualifier)` (999 today) ← mapping `(source, raw_label bytes)`,
  NOT NULL, with `plankton` stored per mapping and `living` derived. [DM, CW, ST, ML]
- **M3 Stable opaque identifiers** for concepts and tree nodes (never the bare name, never an authority id:
  369–500 blanks, no coverage for freshwater taxa, KI-13 reuse); uniqueness on (name, rank, parent); homonyms
  distinct; species as a node/binomial with the epithet derivable. [DM, ST, ML]
- **M4 One loader and a consumer API that keeps every consumer working.** `load_taxonomy()`;
  `build_taxonomy_lookup()` kept as an alias returning the identical dict; a legacy wide `rows()` view for
  sankey, the verifiers and the builders; `render_wide_csv() -> bytes`; the published projection; the
  label-string projection; the legacy CSV still accepted as *input* so the ten fixture-writing test files and
  the Hydra key `taxonomy_csv_path` keep working; new files under the package directory so they ship in the
  wheel. Published-dataset path changes by fewer than 30 lines. [CM]
- **M5 Safe, isolated, idempotent writers.** Per-source partition or key-addressed rows so a builder for source
  X is physically unable to touch source Y; one write API, dry-run by default, never blanking an id without an
  explicit flag, printing a cell-level change list; canonical serialisation so hand edits and tool output
  produce the same bytes; concurrent add-source PRs do not conflict. [CW, CM, DM]
- **M6 Validation on every commit, network-free**, making the following *impossible by construction*: two
  lineages for one concept; two id tuples for one concept outside the enumerated legacy override; a mapping
  without a concept; a rank without its ancestors; a repeating-group id cell; a float-serialised integer; one
  name in two rank slots; a duplicate `(source, raw_label)`. And *detected*: vocabularies; lowercase canonical
  names; coverage of every registered source's frozen class-dir list with no hard-coded row counts; exact ids
  shared by two concepts; the horizontal `Raw_Labels → one concept` check across sources as a waivable lint
  (PR #35, KI-31); donor rank ≤ label rank; bucket labels never mapped to taxa; a missing published column
  raises instead of nulling 17.4 M rows. [DM, CW, ST, CM]
- **M7 Deterministic label strings and vocabularies as data.** A pure projection reproduces today's 849 / 598
  strings in order; label vocabularies (`v1_taxpath`, `proposed_label`, taxpath+qualifier, rank cuts,
  `root_class`, `living`) are versioned data files read by the `ClassLabel` builder, never emergent from code.
  [ML, CM]

**SHOULD**

- **S1** Arbitrary-depth lineage with rank as a node attribute from an open vocabulary (incl. unranked/clade),
  deterministic projection to the seven legacy slots; the 12 not-deepest-node concepts stay distinct.
- **S2** Typed identifier crosswalk: one row per (concept, authority, id) with relation (exact / broader /
  synonym_of), authority rank and status, checked_on, confidence; legacy `ecotaxa_ID` and modern EcoTaxa ids as
  separate authorities; register scope explaining expected blanks.
- **S3** An "according to" (classification source + snapshot date) on each parent link so alternative
  classifications coexist, with one designated projection backbone.
- **S4** Provenance per assertion (method/rule, donor, evidence pointer, curator, date, status), persisted for
  every builder decision; waivers keyed by stable ids and expressible as per-record status so "documented
  defect" and "fixed" are the same representation with a release flag.
- **S5** Explicit reasons for absent ids (`not_in_authority`, `unresolved`, `blanked_too_coarse`,
  `blanked_wrong_taxon`); `unqualified` as an explicit value.
- **S6** Orthogonal descriptor facets (life stage, body part, zooid/colony form, condition, association) with
  controlled vocabularies; `root_class` / `qualifier` / `plankton` derivable with a per-mapping override that
  reproduces frozen values.
- **S7** Non-living, morphological, informal and unresolved classes as first-class nodes with a `node_kind`
  (EcoTaxa's P/M pattern, "no phylogenetic node under a morphological one" enforced).
- **S8** Synonymy and open nomenclature: `name_as_labelled`, an identification qualifier vocabulary (sp., spp.,
  cf., aff., indet., morphospecies id), the precision rank actually asserted.
- **S9** Diff-reviewability: one fact per line, stable sort, narrow records, an optional keyed CSV diff/merge
  driver, a CI-posted semantic change summary; human reports derived from the machine ledger.
- **S10** Rename / merge / split as first-class commands keeping ids stable; every vocabulary release with a
  machine-readable diff and an `old_id → new_id` crosswalk for released `id2label` / `cls_num_list`.
- **S11** The authority snapshot keyed to the distinct-id set, harvested incrementally, so label and lineage
  edits need no network and no snapshot diff.
- **S12** Coverage helpers (`has`, `labels_for`, `datasets`) for `pz_planktonzilla` pre-flight.
- **S13** Hierarchy and captions as data: ancestor closure from parent links; caption templates whose default
  reproduces today's bare rank string; Domain and binomial display names as node data.
- **S14** A curator runbook generated from the tool and an end-to-end test on a synthetic 50-class source.
- **S15** A declared normal form for canonical names, raw labels exempt.

**MAY**: export adapters to ColDP / Darwin Core taxon core, NCBI taxdump shape, OBO-Graph or SKOS (derived,
never canonical); GBIF backbone key and ITIS tsn as extra crosswalk authorities for freshwater taxa; OBO/ENVO/
UBERON IRIs as optional exact-match annotations on facet terms (only 18 of 27 terms have targets).

---

## 5. Prototype evidence: the CSV already decomposes losslessly

A 90-line script (`csv` module only, no repository change) split the committed file into three tables and
regenerated it:

| table | key | columns | rows |
| --- | --- | --- | ---: |
| concepts | `proposed_label` | Kingdom…Species, five ids | 907 |
| mappings | `(Dataset, Raw_Labels)`, in file order | `proposed_label`, `root_class`, `qualifier`, `plankton`, `living` | 2,358 |
| id_overrides | `(Dataset, Raw_Labels)` | five ids | 13 |

The regenerated CSV is **byte-identical** to the committed one (same sha256). The three tables total 277 KB
against 373 KB. Row order of the mapping table is what makes the regeneration byte-exact rather than merely
value-exact, and the `X.0` id form is reproduced only because the prototype keeps ids as strings: a typed store
needs an explicit legacy render rule. All 13 override rows are FREPJ: six concepts where FREPJ rows are
all-blank while 3–14 other sources carry the full id set, four where only `ecotaxa_ID` differs (never written
for FREPJ by design). The implied tree has 1,297 nodes (Kingdom 5, Phylum 30, Class 65, Order 164, Family 288,
Genus 381, Species 364); 230 concept lineages are also ancestors of other concepts, so "class = leaf" is false
and internal nodes must be labelled.

So the question is not *whether* the table can be normalised without loss, but which container, schema and
tooling the normalised form should use.

---

## 6. Prior art

**Biodiversity standards.** No standard models the thing this CSV actually is, a *source label → concept
mapping* with per-mapping attributes; Darwin Core, ColDP, TCS, WoRMS, ITIS and NCBI model concepts and names
only. The only precedents for a stored mapping are EcoTaxa's per-project `taxo_recast` and Darwin Core's
ResourceRelationship. The right move is therefore to split the CSV into a standards-shaped concept table and a
project-specific mapping table, borrowing standard column *names*. The best shape for the concept table is
ColDP NameUsage (`ID, parentID, status, rank, scientificName, …, alternativeID, accordingToID, remarks`): its
117-value rank enum includes `unranked`, its status enum includes `misapplied`, it has an `accordingTo` for the
classification followed, it allows parent link *and* denormalised rank columns on the same row (which is what
lets the frozen columns coexist with a curated tree), and it ships a Frictionless descriptor. GBIF and CoL
explicitly reject names carrying identification qualifiers, so non-living and morphological classes must stay
out of any published taxon core. Python tooling for these standards is thin (`py-coldp`, `python-dwca-reader`,
`pyworms`, none polars-native); the practical pattern is plain TSV read by polars with standard column names,
and standards used as export targets.

**Schema and validation tooling** (measured on a seven-defect fixture: dangling parent, duplicate id,
float-serialised id, dangling mapping FK, bad vocabulary value, duplicate mapping key, `living ≠ root_class`).
Pure polars + pytest catches 7/7 with zero new dependencies in ~15 lines; `pandera[polars]` 7/7 (10 packages,
FKs as `isin` closures); Frictionless Table Schema 6/7 declaratively plus a 6-line check (37 transitive
packages, but the descriptor doubles as documentation and is what ColDP ships); `dataframely` 4/7 declaratively
(1 package); DuckDB catches 7/7 row-by-row but cannot bulk-load a self-referential FK and has no deferred
constraints; `linkml-validate` on CSV found neither the dangling parent nor the duplicate id (83 packages);
Great Expectations has no polars backend. The KI-12 float ids are a CSV-typing artefact of pandas/HF-datasets
inference, not of the data model (the same column via Parquet or a declared string type is clean). CSV/TSV must
stay the git-canonical form: it is the only format that is line-diffable, spreadsheet-editable and read natively
by polars, HF `datasets` and DuckDB; YAML multiplies the diff surface ~7× (16,512 lines for the mappings) and has
no HF loader; Parquet is a fine *derived* artifact but shows as binary in git. `daff` keyed on
`(Dataset, Raw_Labels)` turned the real 229-row id-backfill commit's 458 changed lines into 232 with cell-level
markers and merged two branches that each appended a source block without conflict.

**ML datasets and the plankton community.** Every mature multi-source effort keeps two layers and publishes the
link between them: verbatim source category and harmonised concept (Nocera et al. 2025's global UVP5 database,
EcoTaxa `taxo_recast`, TreeOfLife's per-source name lookups, the SEANOE "XNet" releases' `taxon_level1` /
`taxon_level2`). The plankton community's actual taxonomy model is EcoTaxa's adjacency-list tree with typed
nodes (P phylogenetic / M morphological), status A/N/D with a rename target, an AphiaID per node and
`child<parent` display names; non-living material is a sibling root, and parts / stages / quality classes are
M-type children of P-type nodes, which maps one-to-one onto `(taxon, qualifier, root_class)`. Fixed seven-rank
columns are legitimate as a *derived* publication and training view (TreeOfLife `catalog.csv`, iNat21 JSON,
Darwin Core denormalised ranks) but nowhere serve as the source of truth; TreeOfLife-10M still shipped
0.1–0.2 % hemihomonym mislabels from name matching, the same failure class as the *Ctenophora* case. Coarse
vocabularies for training are separate versioned mapping tables, not extra columns, and training on fine
classes then regrouping outperformed training on coarse classes. Nobody in the community has published a
reusable, machine-readable cross-instrument mapping schema; a versioned, provenance-bearing one would itself be
a contribution.

**Ontology and graph tooling.** A plain adjacency list plus a ~200-line polars/networkx model (both already
installed) delivers every flexibility and robustness property with zero new dependencies; SKOS/rdflib, OBO/
pronto, LinkML and SSSOM-py are all best kept as *derived exports*. SSSOM's column vocabulary
(`subject_id, predicate_id, object_id, mapping_justification, confidence, author_id, mapping_date`) is the right
shape for the id crosswalk table and costs nothing to adopt as a TSV convention. Keyed by (rank, genus+epithet)
the CSV's rank columns already form a DAG with only four genuine multi-parent nodes; stable node ids plus a
species-as-binomial rule remove most apparent conflicts and a tiny `accordingTo` edge table handles the rest.
No plankton ontology exists in OBO Foundry; only 18 of 27 probed descriptor terms have any OBO target.

---

## 7. Candidate designs and how they were evaluated

Six designs were written independently, each from a fixed angle, against the same requirement list and the
same six running example rows (a species with full ids; a `part_` row sharing its organism's concept; a FREPJ
row with per-row blank ids; the `Harpacticoida` horizontal conflict; a family-level mapping with a legacy
EcoTaxa id; a non-living `fiber`). Each was then scored 1–5 on eight criteria by three judges with different
lenses (lossless reproduction and migration risk; curator ergonomics and flexibility; robustness and toolchain
fit) and attacked by a refuter instructed to disprove its load-bearing claims. Judges and refuters did not
take claims on trust: they re-ran the prototypes against the committed CSV, injected defects into the packages,
merged concurrent add-source branches in git, saved files through a spreadsheet round trip, and measured load
and validation times in the project's own virtual environment. Six of the thirty evaluation agents were still
running when this section was written; their results will be folded in when they land, and nothing below
depends on them.

| | Container | Concept identity | Schema of record | New runtime deps | Effort (self-estimate) | Judge mean (n) | Refuted on a load-bearing claim? |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| A. Normalised table package | 31 CSV/TSV files: concepts adjacency list, identifiers, one mapping file per source, vocabularies, legacy pins; Frictionless `datapackage.yaml` | opaque `pz:NNNNNN` + display-name column | Frictionless descriptor (dev) + polars rule module | none | 20 d | 3.79 (3), all three "recommend" | no |
| B. Darwin Core / ColDP checklist | 35 TSV files with DwC term names, SSSOM crosswalk, DwC-Identification-shaped mapping files, `category.tsv`, `release/v1.0/` ledger | readable slug `pzt:abylopsis-tetragona` | Frictionless `datapackage.json` (dev) + polars rule module | none | 18 d | 3.83 (3), "viable" | yes, on workflow guarantees (see below) |
| C. Hierarchical YAML documents | one taxon tree per kingdom + one YAML document per source; pydantic v2 models | readable slug | pydantic (strict) | none | 25 d | 3.71 (3), viable / viable / recommend | yes |
| D. LinkML schema + SKOS/SSSOM | LinkML schema → generated pydantic, JSON Schema, docs; five TSV tables + legacy layer | opaque `PZC:` / `PZT:` numbers | LinkML (dev, +47 packages) + polars integrity module | none | 12.5 d | 3.56 (2), "viable" | pending |
| E. Embedded SQLite | `schema.sql` (20 STRICT tables, 10 triggers, 6 views) as the model; per-table TSV dumps in git; every reader builds an in-memory database | readable `rank:name` ids | SQL DDL | none (stdlib `sqlite3`) | 19 d | pending | pending |
| F. Hardened wide CSV (baseline) | one 20-column CSV, canonical order, integer ids, exceptions file, legacy-order manifest | the `proposed_label` string | Table Schema JSON executed by a hand-written validator | none | 11.5 d | pending | pending |

All six reproduce the committed CSV byte-for-byte from their own store. I re-executed the prototypes of A, B
and D myself (identical sha256); C's converter was not left on disk but its derivation was re-implemented from
the design text by two judges and the refuter, all three obtaining the committed bytes; E and F report the
same result and their judges are still running. The lossless requirement M1 therefore does not discriminate
between designs. Everything else does.

### 7.1 What the evaluation established

**The designs converged.** Five of the six (all but the baseline) independently arrived at the same shape:
a parent-linked concept table with an open rank vocabulary and a stable id that is not the name; one external
id per line in SSSOM-shaped columns; one mapping file per source keyed by the byte-exact raw label and carrying
`root_class`, `qualifier`, `plankton` and provenance; vocabularies as data, including a `legacy_slot` column on
the rank table that turns the seven-column projection into a lookup; a small enumerated legacy layer (a
row-order manifest, the 13 FREPJ id overrides, and a handful of concept or cell pins) that lets a clean model
reproduce the frozen bytes; and a polars rule module for the checks no table schema can express. They differ
in container, id style, and which tool is the schema of record. That convergence is itself a result: the data
model is not in doubt.

**"Impossible by construction" is narrower than every design claimed.** In the table packages only the
storage split is structural: two lineages for one concept, a mapping carrying its own lineage or ids, a
repeating-group id cell, a builder touching another source's file. A dangling parent, a rank gap, a float id,
a bad vocabulary value, a name in two slots, a mapping with no concept, or a duplicate key placed in the wrong
source file are all caught by a validator, and only when the validator runs. Two over-claims recur across
designs and must be corrected in whichever is adopted: Frictionless 5.19 parses `uniqueKeys` but does not
enforce it (an injected duplicate `(name, rank, parent)` concept was reported valid), and a second `exact` id
for the same concept and authority is schema-legal in every SSSOM-shaped table and silently dropped by the
exporter. Both are three-line polars checks.

**Concurrent add-source PRs conflict in git on any shared append-only ledger.** Two branches each appending
a new concept, a new identifier and a new source stanza conflicted on `concepts.csv`, `identifiers.csv` and the
descriptor in design A; on `source.tsv` in design B; on `taxonomy.yaml` and `taxa/animalia.yaml` in design C;
on `concepts.tsv` and `taxa.tsv` in design D even with disjoint id ranges. Only the per-source mapping file is
conflict-free by construction. A committed `.gitattributes` with `merge=union` on the append-only ledgers plus
a duplicate-id rule in the validator resolves it, but no design can honestly say "never conflicts".

**Migration id minting must be idempotent.** Design A's script mints ids in sorted-path order; re-running it
after inserting one row changed the meaning of 1,342 of 1,351 ids while every foreign key still resolved. The
migration must seed ids from the committed concept table and mint only for new nodes, and the three legacy
builders must be disabled from the first migration commit until the write API exists.

**The pin layer must also freeze mapping attributes.** Design A's pins freeze the concept and its ids but not
`root_class`, `qualifier` or `plankton`, so a curator cannot commit a life-stage or KI-10 correction before a
release without turning the golden test red (verified: one qualifier edit changed line 1,557 of the render).
Three nullable columns on the mapping pin fix it.

**The golden test must hash the frozen prefix, not the whole render.** Design B's gate hashed the full
rendered file, so adding a 50-class source turned it red and crashed its error reporter; the frozen-prefix hash
matched throughout. The right gate is "render equals the committed generated CSV" during coexistence, plus the
existing first-1,486-line pin, plus a per-release sha over the keys in the release's own row-order manifest.

**Spreadsheet round trips break booleans and dates.** Excel and LibreOffice write `TRUE` for `true`; design
A's prototype exporter then silently emitted `plankton=False` for all 120 zooscan rows, polars accepted `TRUE`,
Frictionless rejected it, and design D's pydantic layer coerced it while its polars loader rendered it false.
Booleans must be normalised by the formatter and consumed through one typed loader that raises on anything but
`true` / `false`; dates must be strings if spreadsheet editing is to be supported at all.

**A partially mapped source must be committable.** Design B could not: 150 blank rows produced 150 errors
and a renderer crash. A `draft` status that the renderer and the coverage rule skip is required for any source
larger than one sitting.

**YAML costs more than it gives here.** Design C is the best-specified of the six on paper (its legacy
layer, strict pydantic models and duplicate-key loader were praised and borrowed by judges of other designs),
but the migrated ledger is 18,900–20,600 lines against 2,359, there is no spreadsheet view, comments are
forbidden, the lineage is invisible on the mapping line, and as written the ledger fails its own validator (a
label regex that rejects `invertebrate_larvae`, a rank-gap rule whose escape hatch is not a schema field, a
bucket rule that fires on three frozen rows). Its regenerated label-vocabulary file renumbered 537 of 599 class
ids on the first data fix, which is exactly what a released vocabulary must never do.

**Standards vocabulary is worth adopting; a standards format is not.** Design B's ratified Darwin Core term
names and SSSOM columns scored highest on standards alignment, were the easiest for a biologist to read
(`grep Bosmina taxon.tsv` finds `pzt:bosmina-fatalis`), and produced a DwC-A archive that `python-dwca-reader`
read back. But it stores the seven frozen slots redundantly on all 1,305 taxon rows (needing a `reproject`
command and a drift rule), has no home for per-mapping confidence, and its 80 KB generated descriptor repeats
the mapping schema 21 times. Nothing in the standards accommodates the 303 category-linked mappings or the
qualifier vocabulary, so a GBIF export is by construction a subset.

**LinkML's schema forced a wrong statement into the data.** Design D's concept-kind enumeration had no value
for a living class without a tree node, so its migration typed `egg`, `larvae`, `eukaryota`, `zooplankton`,
`spore`, `cyst` and `mix` as non-living: 45 `root_class=living` mappings pointed at non-living concepts. Its
closed enums and classes also made the most common biologist extensions (a new life stage, a per-mapping
confidence, a new authority prefix) engineer-only acts requiring schema regeneration from a 47-package dev
group, and its validator missed a dangling parent and a duplicate id on CSV data. The parts of D that carry
value at runtime (per-source TSVs, SSSOM columns, the polars integrity module, the legacy renderer) do not need
LinkML at all.

**SQLite is the strongest semantic model and the weakest git model.** Design E's triggers make rank
monotonicity, closure maintenance, cycle prevention and "a name never repeats in its own lineage" fail at insert
time, and its views prove the legacy CSV, the lookup and the label strings as SQL. But the constraints hold only
for what is loaded: the TSV dumps in git can be invalid until `build()` rejects them, so on the commit path it
is a validator like the others, with a 291-line DDL that biologists cannot review, GLOB instead of regular
expressions, and parent-first import ordering for the GUI path.

**The baseline is cheap, honest, and fails the blocking requirements.** Design F costs 11.5 days and zero
dependencies, turns every audited anomaly into a detected one, and keeps one spreadsheet. By its own coverage
table it cannot meet M2, M3, S1, S2, S3, S6, S7, S8 or S13: lineage stays 2.6× redundant, the name stays the
identity, a new id on `calanoida` is still a 20-row diff, and homonyms still cannot coexist. It is the right
comparison point and the wrong destination; its validator rule set and its exceptions-file idea are worth
keeping.

### 7.2 Ideas judged worth keeping regardless of container

- A frozen-release layer as data: `row_order.tsv` (physical order stored, never derived, since no sort key
  reproduces the 1,485-row prefix), per-mapping id overrides with a mandatory reason (13 rows), per-concept or
  per-cell pins for KI-8 and KI-9 (5 concepts or 49 cells), and the release sha. Deleting a pin line is the
  fix; bumping the sha is the versioned act.
- `project7()`: fill the seven legacy slots only from ancestors whose rank has a `legacy_slot`; species stored
  as the binomial node with the epithet derived (zero violations over 364 species, tautonyms included).
- The 12 not-deepest-node concepts as real nodes with true ranks (infraorder, suborder, infraclass, tribe) or
  `unranked`, so the training-string collision is an explicit projection artefact rather than data loss.
- Homonyms as distinct nodes (`ctenophora` phylum vs `ctenophora` genus) and the four parent conflicts as two
  nodes each with the classification followed recorded on the node.
- A `kind` vocabulary that includes a value for living classes with no lineage (`egg`, `larvae`, `mix`), with
  EcoTaxa's "no phylogenetic node under a morphological one" rule; parts pointing at the organism plus a
  body-part qualifier; buckets scoped to a parent.
- SSSOM-shaped identifier rows with `skos:exactMatch` / `broadMatch` / `NoTermFound` and separate
  `ecotaxa_legacy` and `ecotaxa` scopes; typed ids that make `135336.0` a validation error; an "absence"
  representation with a reason.
- Explicit `unqualified`; `living` derived from `root_class`; `plankton` stored per mapping.
- A human-readable display name beside the opaque id on every mapping row, asserted equal by CI and filled
  from the name by the formatter, with a stated conflict rule.
- The reproject fixed-point test, the first-divergent-row message in the golden test, and "render equals the
  committed generated CSV" for the whole coexistence period.
- `merged.tsv` (the NCBI `merged.dmp` precedent) plus a rule that an id present in the previous commit and
  absent now must appear there.
- The authority snapshot keyed to the sorted distinct id set instead of the whole-file sha, with incremental
  harvest.
- A CI-posted semantic diff ("N published cells changed on M rows", per cell old → new) computed by rendering
  the legacy view before and after.
- Released label vocabularies as frozen snapshots keyed to a release tag; only new vocabularies are regenerated.

---

## 8. Recommendation

**Adopt a normalised plain-text table package (design A), with Darwin Core and SSSOM column names and the
category split from design B, the strictness and pin-layer lessons from designs C and D, and the refuters'
corrections applied.** Concretely:

```
planktonzilla/planktonzilla_dataset/taxonomy/
├── datapackage.yaml           Frictionless Table Schema descriptor (types, PK, FK, enums, patterns); also the
│                              source registry, one stanza per source in registry order
├── taxon.tsv                  one row per node: taxonID, parentNameUsageID, taxonRank, scientificName,
│                              taxonomicStatus, acceptedNameUsageID, nameAccordingTo, kind, taxonRemarks
├── identifier.tsv             one external id per line: subject_id, predicate_id, object_id (CURIE),
│                              mapping_justification, mapping_date, author_id, confidence, comment
├── mappings/<dataset>.tsv     one file per source: datasetID, verbatimIdentification (byte-exact class dir),
│                              taxonID, concept (display name), root_class, qualifier, plankton, status,
│                              method, donor, identifiedBy, dateIdentified, identificationReferences,
│                              confidence, remarks
├── vocab/rank.tsv             open rank vocabulary with ordinal + legacy_slot
├── vocab/qualifier.tsv        the 14 terms + unqualified, legacy_value, default lifeStage/bodyPart facets
├── vocab/root_class.tsv       living / detritus / artefact / inert + the derived living flag
├── vocab/authority.tsv        scope, id pattern, url template, multi-valued flag, legacy column
├── merged.tsv                 retired ids → replacement, date, reason
├── waivers/*.tsv              tree, horizontal (KI-31), authority: the current JSON ledgers re-keyed by id
└── release/v1.0/              legacy_row_order.tsv (2,358 keys), legacy_overrides.tsv (13 id rows + 5 concept
                               pins, with root_class / qualifier / plankton columns), sha256 over those keys
planktonzilla/planktonzilla_dataset/planktonzilla_taxonomy.csv   GENERATED until the sha pins are retired
.gitattributes                 merge=union on taxon.tsv, identifier.tsv, merged.tsv; eol=lf on *.tsv
```

Why this and not the alternatives, in one line each. TSV/CSV is the only container that is line-diffable,
spreadsheet-editable and read natively by polars, HF `datasets` and DuckDB; design C's YAML is eight times the
lines with no table view, and design E's SQL model is invisible in git. A normalised store is required by M2,
M3, S1, S2, S3, S7 and S13, which design F cannot meet. Darwin Core and SSSOM *names* cost nothing and make the
concept table publishable and familiar, while a Darwin Core *format* would exclude 303 mappings and 44
categories by design B's own admission. LinkML's generators are attractive, but its validator missed structural
defects, its closed schema forced wrong data, and it brings 47 packages. Frictionless is kept as a dev-only
descriptor validator because its row-numbered messages and PK/FK/enum/pattern coverage came free in every
measurement, but the polars rule module is the enforcer and must re-implement the descriptor's constraints so
that pre-flight on a laptop without Frictionless gives the same verdict as CI.

Decisions taken in the recommendation (each maps to an open decision in §10):

- Identifiers are opaque `pzt:` / `pzc:` CURIEs with a zero-padded number (text in spreadsheets, sortable,
  never renamed), *and* every mapping row carries the display name so diffs stay readable. Readable slugs were
  rejected because a rename invites "fixing" the id.
- `plankton` on the mapping row; `living` derived; `unqualified` explicit; `kind` includes a value for living
  classes without a lineage.
- The four parent conflicts stay as two nodes each with `nameAccordingTo` recording the classification
  followed; a backbone switch is a later versioned re-parenting release, not a flag flip.
- Legacy `ecotaxa_ID` is the authority `ecotaxa_legacy`, never backfilled; modern EcoTaxa ids are `ecotaxa`.
- The training vocabulary stays defined on the seven-slot rank string for v1.0 as a frozen snapshot; a
  concept-keyed vocabulary is a new versioned file, not a replacement.
- The Hydra key `taxonomy_csv_path` is kept and accepts a wide CSV or the package directory; the loader keeps
  a wide-CSV backend so the ten fixture-writing test files and the deprecated entry points are untouched.
- The wide CSV stays committed and bundled as a generated artefact with a "not stale" test until the
  golden-diff gate against the Hub artefact exists, so the published path never changes during the migration.

Corrections from the refuters that are part of the recommendation, not optional:

- an id-preserving migration (seed from the committed `taxon.tsv`, mint only new paths, run once) with the
  three builders' write paths disabled in the same commit;
- per-`(subject, scope)` uniqueness for the four single-valued published authorities, and
  `(scientificName, taxonRank, parentNameUsageID)` uniqueness in the polars module, because Frictionless does
  not enforce either;
- the lineage-name-repeat check scoped to legacy-slot ranks (a `bacteria` domain node above the `bacteria`
  kingdom would otherwise fire 30 times);
- booleans normalised by `fmt` and read through one typed loader that raises on anything but `true` / `false`;
  dates stored as strings;
- a `draft` status skipped by the renderer and the coverage rule, so a partially mapped source can be
  committed;
- a stated conflict rule for `taxonID` vs the display name (blank id → resolve by name; both set and
  disagreeing → hard error, never overwrite);
- `merge=union` on the append-only ledgers plus a duplicate-id rule, and an explicit note that a shared
  descriptor stanza may still need a one-line manual resolution when two sources land at once;
- the golden gate as "render equals the committed generated CSV" plus the first-1,486-line pin plus a
  per-release sha over the release's own row-order keys, never a whole-file hash that a new source invalidates.

What the recommendation does not deliver, on purpose: orthogonal facets beyond `lifeStage` and `bodyPart`
columns defaulted from the qualifier vocabulary (S6 is a follow-up once the vocabulary is agreed); a name-usage
table for synonyms (S8: `acceptedNameUsageID` + `taxonomicStatus` cover the three known synonym pairs; open
nomenclature stays in `identificationQualifier`); export adapters (Y1: the concept table already has the ColDP
NameUsage shape, so each is one polars write later); and any change to the published dataset.

---

## 9. Migration plan

Ordered so that the golden gate exists before any consumer moves and the published path never changes. Effort
figures are the panel's own; the refuters judged every estimate a floor (a 937-line builder and a 1,042-line
resolver are being replaced), so treat the total as four to six weeks of one engineer.

1. **Golden gate and one-shot migration (≈4 days).** Write the id-preserving migration script, generate the
   package from the committed CSV, and land the loader, model and legacy renderer with tests that assert:
   rendered bytes equal the committed CSV; the first 1,486 lines equal the sha pin; the 16-column lookup built
   from the *model* (not from the rendered bytes) equals `build_taxonomy_lookup` in value and Python type on all
   2,358 keys; the 849 / 598 label strings in order; the reproject fixed point. Disable the three builders'
   write paths in the same commit. Nothing else changes.
2. **Schema of record and validation (≈3 days).** Commit the descriptor, the polars rule module including the
   refuters' additions, the seven-defect fixture package, a Frictionless CI step in the dev group, and
   adjudicate the day-one lint volume (about 168 coarse-id and 20 horizontal findings) into the waiver tables
   re-keyed by stable ids.
3. **Switch every reader to the loader (≈2 days).** `build_taxonomy_lookup` becomes an alias
   (`generate_planktonzilla.py:105-172` deleted), the copy in `frepj_validate.py` is deleted,
   `check_taxonomy_csv` is rewritten on the loader's pre-flight helpers, and `sankey`, `verify_taxonomy_ids`
   and PR #35's checker read `rows()`. The wide CSV stays committed with a staleness test.
4. **Write side and CLI (≈5 days, the risky step).** `upsert_source`, `set-mapping`, `add-taxon`, `add-id`,
   `rename` / `merge` / `split` with tombstones and automatic pins, `fmt`, `diff`, `release`; port the three
   builders (2,646 lines) onto it; retire `extract_taxon_ids.py`; end-to-end test on a synthetic 50-class
   source; re-express the Tara Pacific "rebuild equals committed block" test on records.
5. **Provenance and waivers as data (≈2 days).** Backfill `method` / `donor` for all 600 Tara Pacific and 229
   FREPJ decisions from the existing reports; seed `broadMatch` rows from the authority findings; derive the
   Markdown reports from the ledger.
6. **Label vocabularies as data (≈1.5 days).** Write the frozen `v1.0_taxpath` snapshot (598 classes plus the
   empty class) and the 21-source `v1.2_taxpath`; `gen_planktonzilla_only_plankton` reads the file; a test
   proves it equals today's `sorted(set())` expression.
7. **Authority snapshot re-keyed to the id set (≈1.5 days; needs the maintainers' HARDEN decision).**
8. **Review tooling, docs, retirement switch (≈1.5 days).** The CI semantic-diff comment, the generated
   curation runbook, README and KNOWN_ISSUES updates, and the documented switch that stops committing the wide
   CSV once the golden-diff gate against the Hub artefact exists.

Only after step 8 does correcting a frozen defect become a one-line, reviewable, versioned act: delete the pin
row, run `release`, and the diff shows exactly the published rows that move.

---

## 10. Decisions for the maintainers

Each with the default the recommendation assumes.

1. **Id style**: opaque numbered CURIE with a display-name column (default) vs readable slug.
2. **Frozen-defect policy**: keep the pin layer until a golden diff against the Hub artefact exists (default),
   or correct the 12-row *Ctenophora* homonym, Harpacticoida / Creseidae and KI-10 now as a v1.1 data release.
3. **Unpublished blocks**: reproduce the Tara Pacific Harpacticoida / Creseidae copies verbatim (default) or
   correct them before their first publication, since nothing pins them yet.
4. **Authority snapshot key**: re-key to the id set with incremental harvest (default) or keep the whole-file
   sha under HARDEN-01/02.
5. **Frictionless in the dev group** (default; 11 packages on top of today's environment, CI +2 s) or the
   polars module alone with the descriptor as documentation.
6. **Row order for post-freeze sources**: registry order then raw-label bytes (default) vs append order.
7. **Write-side actor**: both paths supported (default: flat TSV for biologists, `upsert_source` for
   builders); if only engineers write, the display-name column and the spreadsheet normalisation can be
   dropped.
8. **Next training vocabulary**: keep the rank-string rule (default, 849 classes) or move to concept identity
   (861 classes, keeping brachyura / achelata / gammaridea / cirripedia distinct) in a versioned release with a
   crosswalk.
9. **Parent conflicts**: keep both placements with `nameAccordingTo` (default) or resolve to WoRMS now
   (changes about 50 projected rows).
10. **Life-stage vocabulary** for the `lifeStage` facet: Darwin Core / BODC S11 terms (default) vs EcoTaxa's
    tokens vs a project list.
11. **Wikidata**: keep as a crosswalk authority row (default) or demote to a derived link.
12. **Retire `extract_taxon_ids.py`** (default: retire; its output is `identifier.tsv`), and rebase PR #35's
    checker onto the loader after it merges (default) rather than before.
