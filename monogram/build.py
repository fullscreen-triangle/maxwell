"""
Monograph build script.
Extracts body content from each source paper, strips preambles and frontmatter,
resolves relative \\input paths, and writes clean section files for inclusion.
"""

import re
import os

BASE = r"c:\Users\kunda\Documents\physics\maxwell"
OUT  = os.path.join(BASE, "monogram", "sections")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Papers in monograph order
# Each entry: (slug, relative_path_from_BASE)
# ---------------------------------------------------------------------------
PAPERS = [
    # Part I
    ("p01_observation_boundary",
     r"docs\foundation\observation-boundary\properties-of-observation-boundary.tex"),

    # Part II
    ("p02_gibbs_paradox",
     r"docs\foundation\gibbs-paradox\categorical-resolution-gibbs-paradox.tex"),
    ("p03_maxwell_demon",
     r"poincare\publication\maxwell-demon\maxwell-demon-paradox.tex"),
    ("p04_kelvin_paradox",
     r"poincare\docs\kelvin-paradox\kelvin-paradox.tex"),

    # Part III
    ("p05_categorical_state_counting",
     r"thermodynamics\sources\categorical-state-counting.tex"),
    ("p06_categorical_thermodynamics",
     r"thermodynamics\sources\categorical-thermodynamics.tex"),
    ("p07_categorical_cryogenics",
     r"thermodynamics\sources\categorical-cryogenics.tex"),

    # Part IV
    ("p08_ion_thermodynamic_regimes",
     r"thermodynamics\sources\ion-thermodynamic-regimes.tex"),
    ("p09_equations_of_state",
     r"poincare\docs\equations-of-state\partition-based-equations-of-state.tex"),

    # Part V
    ("p10_ideal_gas",
     r"conjecture\publication\ideal-gas\ideal-gas-laws.tex"),
    ("p11_single_particle",
     r"conjecture\publication\single-particle\single-particle-gas-laws.tex"),
    ("p12_gas_computing",
     r"conjecture\publication\gas-computation\gas-computing.tex"),

    # Part VI
    ("p13_ion_observatory",
     r"thermodynamics\sources\quantupartite-ion-observatory.tex"),

    # Part VII
    ("p14_gas_dynamics",
     r"maupertuis\docs\publications\gas-dynamics\gas-dynamics.tex"),
    ("p15_flux_phenomena",
     r"maupertuis\docs\publications\flux-phenomena\flux-phenomena.tex"),
]

# ---------------------------------------------------------------------------
# Patterns to strip from body
# ---------------------------------------------------------------------------
STRIP_PATTERNS = [
    r"\\maketitle\s*",
    r"\\begin\{abstract\}.*?\\end\{abstract\}",
    r"\\tableofcontents\s*",
    r"\\clearpage\s*",
    r"\\newpage\s*",
    # revtex frontmatter
    r"\\preprint\{[^}]*\}\s*",
    r"\\pacs\{[^}]*\}\s*",
    r"\\keywords?\{[^}]*\}\s*",
    # author/affiliation blocks (revtex style)
    r"\\author\{[^}]*\}\s*\\email\{[^}]*\}\s*\\affiliation\{[^}]*\}\s*",
    r"\\author\{[^}]*\}\s*\\affiliation\{[^}]*\}\s*",
    r"\\author\[.*?\]\{.*?\}\s*\\affil\[.*?\]\{.*?\}\s*",
    # date
    r"\\date\{[^}]*\}\s*",
    # \twocolumn[...] wrappers (used in some papers for abstract in two-column)
    r"\\twocolumn\[\s*\\begin\{@twocolumnfalse\}.*?\\end\{@twocolumnfalse\}\s*\]\s*",
    # addcontentsline for parts within papers (we regenerate these)
    r"\\addcontentsline\{toc\}\{part\}\{[^}]*\}\s*",
    r"\\addcontentsline\{toc\}\{section\}\{[^}]*\}\s*",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_body(text):
    """Strip everything before \\begin{document} and after \\end{document}."""
    m = re.search(r'\\begin\{document\}', text)
    if not m:
        return text
    body = text[m.end():]
    m2 = re.search(r'\\end\{document\}', body)
    if m2:
        body = body[:m2.start()]
    return body

def strip_frontmatter(body):
    """Remove maketitle, abstract, TOC, and revtex-specific frontmatter."""
    for pat in STRIP_PATTERNS:
        body = re.sub(pat, '', body, flags=re.DOTALL | re.MULTILINE)
    return body

def remap_internal_parts(body):
    """
    Papers that use \\part*{} internally as section dividers
    (e.g. bounded-phase-space, hyperfine-transitions) need those
    demoted to \\subsection* so they don't override the monograph's \\part{}.
    """
    # \\part*{Title} -> \\subsection*{Title}  (starred variant)
    body = re.sub(r'\\part\*\{', r'\\subsection*{', body)
    # \\part{Title}  -> \\subsection{Title}   (numbered variant, rare in bodies)
    # but only if it's in the body — we detect by context
    # (safe to do globally; the monograph master uses \part itself)
    body = re.sub(r'\\part\{', r'\\subsection{', body)
    return body

def resolve_inputs(body, paper_dir):
    """
    Replace relative \\input{sections/foo} with absolute paths
    so they still work when included from a different directory.
    """
    def replacer(m):
        rel = m.group(1).strip()
        # Add .tex if missing
        if not rel.endswith('.tex'):
            rel = rel + '.tex'
        abs_path = os.path.join(paper_dir, rel).replace('\\', '/')
        return '\\input{' + abs_path + '}'
    # Match \input{...} that don't look like absolute paths already
    body = re.sub(r'\\input\{([^/][^}]*)\}', replacer, body)
    return body

def collect_bibs(text):
    """Extract bibliography file names from \\bibliography{...}."""
    m = re.search(r'\\bibliography\{([^}]+)\}', text)
    if not m:
        return []
    return [b.strip() for b in m.group(1).split(',')]

def make_bib_paths(bibs, paper_dir):
    """Convert relative bib names to absolute paths."""
    paths = []
    for b in bibs:
        if not b.endswith('.bib'):
            b = b + '.bib'
        abs_p = os.path.join(paper_dir, b)
        if os.path.exists(abs_p):
            paths.append(abs_p.replace('\\', '/'))
    return paths

# ---------------------------------------------------------------------------
# Process each paper
# ---------------------------------------------------------------------------

all_bib_paths = []

for slug, rel_path in PAPERS:
    full_path = os.path.join(BASE, rel_path)
    paper_dir = os.path.dirname(full_path)

    if not os.path.exists(full_path):
        print(f"WARNING: not found: {full_path}")
        # Write placeholder
        out_path = os.path.join(OUT, slug + '.tex')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"\\section*{{[Paper not found: {rel_path}]}}\n\n")
        continue

    with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()

    body = extract_body(raw)
    body = strip_frontmatter(body)
    body = remap_internal_parts(body)
    body = resolve_inputs(body, paper_dir)

    # Collect bibs
    bibs = collect_bibs(raw)
    bib_paths = make_bib_paths(bibs, paper_dir)
    all_bib_paths.extend(bib_paths)

    # Write section file
    out_path = os.path.join(OUT, slug + '.tex')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(body.strip() + '\n')

    print(f"OK  {slug}  ({len(body)} chars)")

# ---------------------------------------------------------------------------
# Write bibliography list to a helper file
# ---------------------------------------------------------------------------
bib_list_path = os.path.join(BASE, "monogram", "bib_list.txt")
unique_bibs = list(dict.fromkeys(all_bib_paths))  # deduplicate, preserve order
with open(bib_list_path, 'w') as f:
    f.write('\n'.join(unique_bibs))
print(f"\nBibliography files ({len(unique_bibs)}):")
for b in unique_bibs:
    print(f"  {b}")

print("\nDone. Section files written to:", OUT)
