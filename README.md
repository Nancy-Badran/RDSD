# RDSD — synthetic methodology demonstration

Independent project; keep the five-attribute example separate.

## Windows / VS Code
Open this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\run_demo.py
Start-Process .\output\identity_ecosystem.html
```

Already generated: data/profiles.csv (10,000 fictional profiles, 20 binary attributes), config/reference_bn.json (explicit synthetic generator), output/learned_bn.json (learned structure and Laplace-smoothed CPTs), output/validation.json, interactive HTML, poster_graph.svg and poster_graph.png. SVG is recommended for a poster.

## Methodology
1. Define a synthetic reference DAG and explicit CPTs. Binary states simplify the demonstration; they are not empirical demographic categories. Each non-root node has two parents; strong logistic effects of magnitude 1.9 are deliberately used for recoverable dependencies. A few effects have negative signs. No probabilities are calibrated to ITAP.
2. Generate 10,000 profiles via ancestral sampling, NumPy RNG seed 42.
3. Split into 8,000 training / 2,000 test profiles, seed 43.
4. Learn structure from training observations by exhaustive local BIC parent selection under a supplied variable order, maximum two parents. The learner never uses the reference parent lists or CPTs. However, the expert order matches the generating order and therefore helps recovery. This is constrained structure learning, not unconstrained causal discovery.
5. Estimate CPTs with Laplace pseudocount 1. Evaluate mean joint negative log likelihood on held-out profiles against an independent baseline trained on the same training split. Compare edge recovery to the known synthetic reference as a separate simulation check.
6. Export learned DAG to interactive HTML and publication-scale SVG/PNG. Node colors identify domains; dark borders identify the three scenario-sensitive targets. Size is uniform, not a risk estimate.

One strong-effect scenario and one split do not establish generalization or real-world validity. This project implements the offline demonstration only: no real incident history, verification policy, PRS or strategy optimization is included. Do not use the five-variable scripts unchanged with this dataset. Full joint enumeration on 20 binary variables has 2^20 configurations; use an appropriate scalable inference method for later RDSD evaluation.

## Suggested poster caption
Synthetic identity ecosystem learned from 10,000 simulated profiles covering 20 binary attributes. Bayesian network structure is estimated using BIC under an expert-specified ordering and a maximum of two parents per variable; CPTs are estimated on 8,000 training profiles and evaluated on 2,000 held-out profiles. Colors indicate attribute domains, and dark borders mark scenario-sensitive targets. The figure illustrates the offline methodology and does not represent empirical ITAP data or causal relationships.
