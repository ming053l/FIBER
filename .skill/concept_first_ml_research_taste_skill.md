---
name: concept-first-ml-research-taste
description: >
  Use this skill when brainstorming, framing, writing, reviewing, or visualizing
  machine-learning research. It favors concept-first reframing, minimal but
  revealing theory, mechanism-driven experiments, clean explanatory figures,
  and a tight theory–algorithm–system narrative. Apply it to research ideation,
  paper outlines, introductions, method sections, experiment design, figure
  planning, titles, abstracts, and reviewer-facing revisions.
---

# Concept-First ML Research Taste

## Goal

Produce ML research that feels **inevitable after the key insight is stated**.

The target style is not “add another module and show +0.3%.”
Instead:

1. identify an accepted framing, approximation, heuristic, or assumption;
2. expose the exact place where that framing is incomplete;
3. replace it with a cleaner conceptual view;
4. derive a minimal method from that view;
5. show that prior methods become understandable special cases, limits, or approximations;
6. validate the mechanism before chasing benchmark scale;
7. make every figure, theorem, and experiment carry one part of the same argument.

The paper should leave the reader with a reusable concept, not merely a method name.

---

# 1. Research Taste

## 1.1 Prefer a framing shift over an architectural tweak

Strong starting points look like:

> Existing work asks **which X to choose**.  
> The more fundamental question is **how X should constrain / shape / transform Y**.

> Existing work treats phenomenon X as additive / independent / static.  
> The actual object contains interaction / geometry / dynamics that this view hides.

> Existing methods optimize an empirical proxy.  
> Before improving the proxy, ask whether the proxy measures the desired quantity at all.

A good project should be explainable as a change of viewpoint:

**old view → missing structure → new view → larger design space**

Do not begin from:
- “Can we insert module A into model B?”
- “Can we combine method X and method Y?”
- “Can we improve the SOTA number?”

Begin from:
- “What assumption makes the current formulation convenient?”
- “When does that assumption stop being faithful?”
- “What structure is being thrown away?”
- “Can the problem be rewritten so existing methods appear as special cases?”

---

## 1.2 Attack the simplest setting first

When challenging an accepted practice, try to break it in the easiest regime where people expect it to work.

Preferred progression:

**closed-form toy setting → theorem/counterexample → named mechanism → synthetic validation → realistic model**

Examples of useful simple regimes:
- linear regression;
- one linear layer;
- quadratic objective;
- low-dimensional geometry;
- one-step optimization;
- sparse vector/matrix toy problem.

A failure in a simple model is rhetorically stronger than a mysterious failure in a giant neural network.

Use the simple setting to isolate causality, not merely as a warm-up.

---

## 1.3 Name the mechanism

Do not stop at “method X fails.”

Find the mechanism and give it a memorable name.

Good mechanism names describe what happens:
- amplification;
- cancellation;
- under-estimation;
- data regularization;
- effective sparsity;
- projection distortion;
- bias–variance tradeoff;
- approximation error;
- system overhead.

A named mechanism becomes:
1. a subsection title;
2. a theorem target;
3. a figure;
4. an ablation;
5. a sentence reviewers remember.

If the phenomenon cannot yet be named, the analysis is probably not finished.

---

## 1.4 Prefer structure-exploiting algorithms

When designing an efficient method, ask:

> What property does the generic algorithm ignore?

Examples:
- sparsity;
- factorization;
- low rank;
- symmetry;
- locality;
- repeated structure;
- manifold/geometry;
- separability;
- batch/sample asymmetry.

The ideal efficiency story is:

**generic computation is expensive → data/model has structure → exploit structure before approximation → complexity falls for a principled reason**

Avoid “we use CUDA so it is faster” as the main contribution.

System optimization is strongest when it realizes an algorithmic insight that already has a clean complexity argument.

---

## 1.5 Theory should diagnose or organize, not decorate

Every theorem should do at least one of these:

- prove a widely used heuristic can fail;
- isolate the exact term responsible for failure;
- characterize when a method works;
- reveal a tradeoff;
- unify multiple algorithms;
- justify a complexity reduction;
- define a useful design space.

Avoid theorem-shaped appendices that do not change how the reader thinks about the method.

After an important theorem, explicitly write the practical consequence.

Preferred pattern:

**Theorem → one-paragraph interpretation → Takeaway → next design choice**

---

# 2. Paper Narrative

## 2.1 Introduction architecture

Use this five-act structure.

### Act 1 — Establish the real problem

Explain why the problem matters, but move quickly to the technical tension.

Prefer:
> Target data are faithful but scarce; general data are abundant but misaligned.

over:
> Large language models have recently achieved remarkable success...

State the conflict as two desirable properties that cannot both be obtained naively.

---

### Act 2 — Describe the dominant view fairly

Explain why the existing formulation is natural and why people use it.

Do not create a straw man.

The reader should think:
> Yes, that is exactly how the field currently sees the problem.

---

### Act 3 — Reveal what the current framing hides

Use a pivot sentence such as:

> However, this perspective obscures a more fundamental question: ...

or

> A more significant concern lies in the assumption that ...

This sentence should be the intellectual hinge of the introduction.

---

### Act 4 — Introduce the new perspective before the method

First state the conceptual replacement.

Only afterward introduce the algorithm.

Preferred order:

**perspective → mathematical object → prior methods as special cases → new method → implementation**

Not:

**module → loss → implementation → post-hoc interpretation**

---

### Act 5 — Close the loop with reality

If the method is intended for modern ML, address implementation explicitly.

A strong contribution list often separates:

- **Framework** — new conceptual formulation;
- **Method** — algorithm derived from it;
- **Theory** — guarantee/failure/tradeoff;
- **System** — practical scalable realization;
- **Experiments** — mechanism + downstream evidence.

Only include categories that are real contributions.

---

# 3. Section Titles Should Carry the Argument

Prefer argumentative section titles:

- “Pitfalls of …”
- “Why X Fails Even in Linear Models”
- “Violation of the Additivity Assumption”
- “The Dual Perspective of X and Y”
- “Promises of the Adaptive Algorithm”
- “Effective Parameter Sparsity”
- “Scaling to Billion-Parameter Models”

Avoid generic titles when a stronger claim is available:

- “Analysis”
- “Motivation”
- “Method Details”
- “More Experiments”

The table of contents should almost summarize the paper by itself.

---

# 4. Figure Grammar

## 4.1 One figure = one inference

Never ask one figure to simultaneously explain:
- the entire architecture,
- all losses,
- training,
- inference,
- theory,
- and results.

Instead split the reasoning into atomic figures.

The figure should answer one sentence:
> What should the reader understand after looking at this for five seconds?

---

## 4.2 Derive visual complexity progressively

When the method modifies a familiar computation, draw a sequence:

**baseline primitive → structural observation → simplified primitive → final method**

For example:

**dense operator → sparse operator → exploit input sparsity → composed algorithm**

This makes the method appear logically derived rather than invented.

---

## 4.3 Preferred visual vocabulary

Use:
- vectors;
- matrices;
- dots;
- arrows;
- low-dimensional manifolds;
- feasible regions;
- projection arrows;
- small point clouds;
- simple regression lines;
- compact before/after diagrams;
- zoomed insets.

Use very few colors.

Recommended semantic palette:
- neutral gray/black = baseline/context;
- one accent = target/query/important quantity;
- second accent = proposed structure/selected subset/constraint.

Color must encode meaning, not decoration.

---

## 4.4 Put the mathematics inside the picture

Prefer labels such as:

- `g*`
- `Proj_U(g*)`
- `θ_t → θ_{t+1}`
- `P g`
- `selected S`
- `forward`
- `backward`

over long prose outside the figure.

Readers should be able to map symbols from equations directly onto the diagram.

---

## 4.5 Caption = conclusion

Bad:
> Figure 3: Overview of our method.

Better:
> Figure 3: Sparse projection exploits nonzeros in the input, making projection cost scale with effective gradient sparsity.

A caption should tell the reader what the figure proves or explains.

---

## 4.6 Toy figures should expose failure modes

For theoretical or diagnostic papers, construct a toy example where:
- the baseline makes a visibly plausible choice;
- the true optimum is visibly different;
- one geometric/statistical quantity explains why.

If helpful, use:
- numbered samples;
- direct labels on points;
- dashed test/query line;
- an inset zoom;
- two competing fitted models.

The toy figure should be reproducible from the theorem assumptions.


## 4.7 Figure production stack

Use the drawing tool that matches the **epistemic role** of the figure.

### Empirical / quantitative figures → Python + Matplotlib

Use Matplotlib for:
- scatter plots;
- toy regression;
- ablation curves;
- runtime / memory plots;
- Pareto frontiers;
- sensitivity analysis;
- histograms;
- inset zooms;
- benchmark comparisons.

Preferred characteristics:
- vector export (`.pdf` or `.svg`);
- LaTeX-compatible math labels;
- minimal spines/grid;
- direct annotations when possible;
- no decorative gradients;
- few marker shapes;
- one visual variable per semantic role.

A quantitative figure should look like a scientific instrument, not a dashboard.

Recommended workflow:

```text
experiment output
    ↓
Python script
    ↓
Matplotlib
    ↓
PDF/SVG
    ↓
LaTeX paper
```

Keep the plotting script in the repository so every figure is reproducible.

---

### Conceptual / theoretical figures → TikZ or equivalent vector drawing

Prefer TikZ/PGF when the figure contains:
- vectors and projections;
- feasible sets;
- manifolds;
- gradients;
- matrices;
- sparse/dense operators;
- low-dimensional geometric arguments;
- mathematical transformations;
- small algorithm derivations;
- symbolic labels that must exactly match the paper.

Good conceptual diagrams often use only:

```text
points
arrows
curves
sets
matrices
simple boxes
equation labels
```

Avoid generic neural-network icons unless the architecture itself is the contribution.

A conceptual figure should behave like a **visual theorem**:
every object must correspond to a mathematical concept in the text.

Recommended workflow:

```text
paper claim
    ↓
minimal geometric abstraction
    ↓
TikZ / vector drawing
    ↓
PDF
    ↓
LaTeX paper
```

---

### Figma / Illustrator / PowerPoint are secondary tools

Use a free-form vector editor only when:
- irregular shapes are difficult to express cleanly in TikZ;
- the diagram needs manual layout iteration;
- many non-mathematical graphical objects are involved.

If using Figma / Illustrator / PowerPoint:
- keep shapes flat;
- avoid gradients and shadows;
- export vector graphics;
- typeset important mathematical labels in LaTeX when possible;
- do not let presentation-slide aesthetics leak into the paper.

PowerPoint is acceptable for prototyping, but the final figure should still follow the same scientific visual grammar.

---

## 4.8 Tool-selection rule

Before drawing, classify the figure.

### Type A — Measurement

Question:
> “What did the experiment measure?”

Use:
**Matplotlib**

Examples:
- accuracy vs runtime;
- error vs rank;
- sparsity histogram;
- toy regression;
- scaling law.

---

### Type B — Mechanism

Question:
> “Why does the phenomenon occur?”

Use:
**TikZ / vector geometry**

Examples:
- amplification;
- cancellation;
- projection distortion;
- feasible-set restriction;
- gradient interaction.

---

### Type C — Derivation

Question:
> “How does the proposed algorithm arise from the observation?”

Use:
**TikZ**

Preferred layout:

```text
standard operation
      ↓
expose ignored structure
      ↓
simplified operation
      ↓
proposed algorithm
```

Do not begin with the final complicated pipeline.

---

### Type D — System

Question:
> “How do components interact in implementation?”

Use:
**TikZ, Figma, or Illustrator**

Keep the number of boxes low.
Separate:
- logical components;
- data movement;
- expensive operations;
- reusable/precomputed quantities.

The diagram should expose the bottleneck or design choice, not merely enumerate modules.

---

## 4.9 Visual style specification

Default visual language:

### Color

Use approximately three semantic levels:

- **black / dark gray** — baseline structure, axes, equations;
- **light gray** — context, inactive objects, background sets;
- **one accent color** — the proposed quantity, query, selected subset, or important vector;
- optionally **one secondary accent** when a real contrast is needed.

Do not assign a new color to every component.

Use color consistently across the entire paper.

For example:

```text
gray   = existing / background / full population
pink   = target / query / important direction
blue   = proposed / selected / constrained quantity
```

The exact hue matters less than semantic consistency.

---

### Typography

- Match the paper's math font.
- Prefer LaTeX-rendered symbols for all mathematical objects.
- Use short labels directly in the diagram.
- Avoid large bold presentation-style headers inside figures.
- Figure text should remain readable at final two-column print size.

If a label requires a full sentence, the figure is probably too complicated.

---

### Lines and shapes

Prefer:
- thin-to-medium strokes;
- rounded or smooth geometric sets;
- simple arrowheads;
- dashed lines only for meaningful distinctions;
- sparse fills with low visual weight.

Avoid:
- drop shadows;
- pseudo-3D;
- glossy boxes;
- thick borders;
- decorative icons;
- unnecessary grid lines.

---

## 4.10 The “visual theorem” test

For a conceptual figure, explicitly map each visual element:

| Visual element | Mathematical meaning |
|---|---|
| point | sample / parameter / state |
| arrow | update / projection / mapping |
| region | feasible set / distribution / constraint |
| dashed line | approximation / counterfactual / reference |
| color accent | target / proposed object |
| matrix sparsity | computational structure |
| distance | error / discrepancy / geometry |

If an object has no precise meaning, delete it.

---

## 4.11 Progressive figure construction

Do not reveal the full method immediately.

Whenever possible, construct the explanation in stages:

### Panel 1 — Familiar baseline
Show what the reader already knows.

### Panel 2 — Missing structure
Highlight the exact thing the baseline ignores.

### Panel 3 — Consequence
Show the resulting failure, inefficiency, or bias.

### Panel 4 — Proposed correction
Introduce only the operation needed to fix it.

This creates the impression:

> “Of course the method should look like this.”

rather than:

> “Why did the authors invent all these blocks?”

---

## 4.12 Reproducibility requirement

All quantitative figures should be regenerated from code.

Preferred repository structure:

```text
figures/
    plot_main_result.py
    plot_ablation.py
    plot_toy_failure.py
    data/
    generated/
```

For TikZ figures:

```text
figures/
    concept_failure.tex
    concept_method.tex
    system_pipeline.tex
```

Do not manually edit numerical plot positions in Illustrator after export unless unavoidable.

The paper source should remain the single source of truth.

---

## 4.13 Caption style

Captions should communicate **the inference**, not the inventory.

Avoid:

> “Overview of the proposed framework.”

Prefer:

> “Restricting the update to the target-induced feasible region prevents general-domain gradients from moving the model in directions unsupported by target data.”

Avoid:

> “Comparison of dense and sparse projection.”

Prefer:

> “Exploiting gradient sparsity reduces projection cost without changing the target geometry represented by the sketch.”

A reader who reads only:
- title,
- abstract,
- figures,
- captions,

should still recover the core paper argument.

---

## 4.14 Figure debugging procedure

When a figure feels weak, debug it in this order:

1. Write its intended conclusion in one sentence.
2. Remove every object not needed for that sentence.
3. Replace prose with mathematical labels.
4. Reduce colors to semantic roles.
5. Check whether the figure should be split into multiple panels.
6. Check whether a toy geometric example would explain the idea better.
7. Check whether the caption states the conclusion.
8. Print or preview at final paper width.

If the core claim disappears at two-column scale, redesign the figure.

---

## 4.15 Default production recipe

When emulating this research style, use:

```text
Empirical plots:
Python + Matplotlib
        ↓
PDF/SVG

Theory / mechanism:
TikZ / PGF
        ↓
PDF

Irregular conceptual illustration:
Figma / Illustrator
        ↓
PDF/SVG
        ↓
LaTeX math labels when needed
```

The goal is not to imitate a specific software package.

The goal is to reproduce the **visual discipline**:

**minimal objects + mathematical semantics + progressive reasoning + one inference per figure.**

---

# 5. Theory-to-Figure Pattern

For each major theoretical result, try this sequence:

1. **Claim**  
   State exactly what common intuition fails.

2. **Closed form / decomposition**  
   Write the quantity in a form where the missing term becomes visible.

3. **Name the term or mechanism**  
   e.g. interaction, leverage, cross-term, projection error.

4. **Toy figure**  
   Show the mechanism geometrically.

5. **Theorem / proposition**  
   Formalize the failure or guarantee.

6. **Takeaway box or bold sentence**  
   Explain what researchers should stop assuming.

7. **Algorithmic implication**  
   Modify the method specifically to address that mechanism.

Never leave a theorem floating without an algorithmic or conceptual consequence.

---

# 6. Experiment Design

## 6.1 Experiments should answer claims, not fill tables

For every experiment, write the question first.

Examples:
- Does the predicted failure mode actually occur?
- Is the gain due to adaptivity or merely more computation?
- Does sparsity survive at realistic model scale?
- Does the proposed approximation preserve the target quantity?
- Where is the accuracy–efficiency frontier?
- Does the new view continue to help outside the toy regime?

If no sentence-level question exists, reconsider the experiment.

---

## 6.2 Preferred experimental ladder

### Level A — Mechanism
Use synthetic or controlled data to isolate the proposed phenomenon.

### Level B — Fidelity
Measure whether the approximation preserves the quantity the paper claims to preserve.

### Level C — Tradeoff
Plot accuracy/fidelity against memory/time/compute/regularization strength.

### Level D — Scale
Demonstrate that the method still works in realistic models/datasets.

### Level E — Downstream consequence
Show that the mechanism matters for the actual application.

Do not jump directly from theorem to a giant benchmark leaderboard.

---

## 6.3 Compare against the strongest relevant abstraction

If the proposed method is about compression, compare compression quality and compression cost.

If it is about attribution fidelity, compare actual/counterfactual influence fidelity.

If it is about data selection, evaluate the effect of the selected data—not merely correlation with another attribution score.

Choose evaluation quantities that match the causal claim.

Be willing to question an established metric if it rewards the wrong abstraction.

---

## 6.4 Expose the frontier, not only the winning point

When a method trades one property for another, show the curve.

Useful axes:
- quality vs runtime;
- fidelity vs compression;
- bias vs variance;
- target performance vs general performance;
- memory vs throughput;
- approximation order vs error.

A well-chosen curve often communicates more research insight than a large table.

---

# 7. Writing Style

## 7.1 Use conceptual contrast

Frequent useful structures:

> X is attractive because ..., **however** ...

> The key observation is ...

> From this perspective, ... arises naturally.

> This suggests a broader design space.

> In contrast, ...

> Interestingly, existing methods arise as special cases ...

> The role of X is central: ...

These phrases should mark genuine logical transitions, not filler.

---

## 7.2 Define roles asymmetrically

When two objects play different roles, say so explicitly.

Example template:

> A determines **what** should be optimized, whereas B determines **how** the optimization is allowed to proceed.

This is stronger than presenting A and B as two symmetric inputs to a pipeline.

Look for such asymmetries—they often contain the paper's conceptual contribution.

---

## 7.3 Reuse one vocabulary throughout

Choose one term for each object and keep it stable.

If the key object is called a “feasible set,” do not later alternate among:
- constraint region;
- candidate space;
- update manifold;
- admissible domain

unless the distinction is mathematically real.

Stable terminology makes abstract work feel simpler.

---

## 7.4 Explain equations with causal language

After a key equation, do not merely paraphrase symbols.

Explain:
- which term causes amplification;
- which denominator creates instability;
- which factor controls runtime;
- which component is estimated from scarce data;
- which approximation destroys structure.

The equation should answer “why,” not only “what.”

---

# 8. Naming and Titles

## 8.1 Method names

Prefer short names that encode the mechanism.

Good pattern:
- [operation/property] + [operation/property]
- acronym that can be pronounced;
- derived variant name that exposes structure.

The name should help recall the method.

Avoid forced acronyms that require an entire sentence to decode.

---

## 8.2 Paper title templates

Prefer titles with a conceptual second half:

> **Method: Mechanism and Consequence**

> **Problem: Challenges, Promises, and Beyond**

> **Method: A [New Perspective] on [Problem]**

> **Phenomenon via [Mechanism]**

> **How Faithful Is X? Error Sources, Remedies, and Practical Guidelines**

A good title tells readers the intellectual object, not only the benchmark task.

---

# 9. Research-Idea Filter

Before committing to a project, score each item from 0–2.

## Concept
- Is there a one-sentence reframing?
- Does it reveal structure hidden by the dominant view?

## Diagnosis
- Can the weakness be shown in a simple setting?
- Can the failure mechanism be named?

## Derivation
- Does the proposed method follow naturally from the diagnosis?
- Can existing methods be interpreted as special cases/limits?

## Evidence
- Is there an experiment that isolates the mechanism?
- Is there a realistic-scale experiment that proves relevance?

## Practicality
- If efficiency is claimed, is the implementation actually faster?
- If scalability is claimed, is full-system overhead measured?

## Memorability
- Can one clean figure explain the core insight?
- Can a reviewer summarize the paper in two sentences?

Interpretation:
- **20–24**: very strong framing; proceed.
- **15–19**: promising; sharpen the mechanism.
- **10–14**: likely method-first; revisit the question.
- **<10**: do not optimize experiments yet; find a better conceptual core.

---

# 10. Figure Review Checklist

For every figure, ask:

- Can a reader understand the point without reading two paragraphs first?
- Does each color have a semantic role?
- Is there only one major message?
- Are mathematical symbols consistent with the text?
- Could 30% of the boxes/arrows be deleted?
- Does the caption state the conclusion?
- If it is a toy example, does it directly instantiate the theory?
- If it is a result plot, does it reveal a tradeoff/mechanism rather than only ranking methods?

If the answer to the deletion question is “yes,” simplify the figure.

---

# 11. Introduction Review Checklist

A strong introduction should contain, in order:

1. the practical/scientific tension;
2. the dominant formulation;
3. the exact hidden assumption or limitation;
4. one sentence beginning the reframing;
5. the mathematical object enabling the reframing;
6. what becomes newly possible;
7. the practical implementation challenge;
8. the evidence that closes the loop;
9. a contribution list organized by intellectual role.

Delete generic field-history paragraphs unless they are necessary for the tension.

---

# 12. How to Use This Skill on a New Project

When given a research idea, do **not** immediately write the paper.

First return:

### A. Current framing
What is the standard way the field thinks about the problem?

### B. Hidden assumption
What approximation, independence assumption, metric, or computational abstraction is taken for granted?

### C. Stress test
What is the simplest model where that assumption can be tested exactly?

### D. Mechanism
If it fails, what term or interaction causes the failure?

### E. Reframing
What alternative perspective captures that mechanism?

### F. Minimal method
What is the smallest algorithm that follows from the new perspective?

### G. Unification
Which existing methods become special cases, limits, or approximations?

### H. Killer figure
Describe one figure that would make the core insight obvious.

### I. Experiment ladder
List mechanism → fidelity → tradeoff → scale → downstream experiments.

### J. Reviewer sentence
Finish:

> “The main reason this paper is interesting is not that it improves X, but that it shows ______ and turns this into ______.”

Only after these ten items are coherent should the paper be drafted.

---

# 13. Anti-Patterns

Reject or rewrite ideas with these symptoms:

- contribution is “A + B + new loss”;
- novelty depends entirely on benchmark gain;
- theory is disconnected from the implementation;
- figures are decorative architecture diagrams;
- every experiment is a large table;
- no simple counterexample or mechanism exists;
- efficiency is only FLOPs, not measured wall-time/memory;
- a new term is introduced for an old idea without a changed viewpoint;
- the introduction says “existing methods have limitations” but cannot name the assumption;
- ablations remove modules but never test the claimed mechanism;
- the paper has many tricks but no sentence-level intellectual thesis.

---

# 14. Desired Final Feel

The finished paper should make the reader think:

> “I had seen all of these ingredients before, but I had not organized the problem this way.”

and then:

> “Given that viewpoint, the proposed method is the natural thing to do.”

That is the target research taste.
