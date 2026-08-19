---
name: creating-question-templates
description: Use when a new problem file is added under interview-questions/ in this repo and a matching Python starter template is needed under templates/.
---

# Creating Question Templates

## Overview

Every problem markdown file `interview-questions/NN_slug.md` gets a matching runnable starter file at `templates/NN_slug.py`. The template implements only the dataset-loading step; everything else is a stub the user fills in during practice.

## When to Use

- A new `interview-questions/NN_slug.md` exists (or is being created) with no matching `templates/NN_slug.py`.
- Not for filling in the actual solution — templates stay unsolved.

## Output Contract

`templates/NN_slug.py` (same number + slug as the source `.md`, `.py` extension), in order:

1. Header comment: one line naming the question, one line pointing back to the source `.md`.
2. Imports: only what `load_data()` actually needs, executable. Below that, a commented-out `# TODO: import whatever else you need, e.g.` block listing likely imports inferred from the md's "Expected Output"/"Tasks" (pipeline, preprocessing, model families, metrics).
3. `load_data() -> pd.DataFrame`: fully implemented, not a stub — loads the real dataset the way the `.md` specifies (e.g. `sns.load_dataset(...)`, `load_iris(...)`) and returns it.
4. One stub function per remaining numbered `Tasks` item in the md (skip the load-data task), named for what it does (`inspect_data`, `split_features_target`, `split_train_test`, `build_pipeline`, `evaluate_model`, `train_alternative_model`, etc.). Each has a `"""TODO: ..."""` docstring paraphrasing that task and a bare `pass` body — nothing else.
   - Merge adjacent tasks into one stub when they're really one logical step (e.g. "build a Pipeline" + "train a baseline classifier" → a single `build_pipeline()`).
   - Skip tasks that ask for prose, not code (e.g. "briefly explain your model choice") — no stub function for these; they get answered inline as comments/writeup when the question is actually solved.
5. `main()`: calls `load_data()`, prints dataset size (`df.shape`) and a sample (`df.head()`), then a `# TODO:` block with the stub calls commented out to show intended wiring. Do not call the stub functions for real — they return `None`, so wiring them up live will crash or silently produce garbage.
6. `if __name__ == "__main__": main()`

## Steps

1. Read the source `.md` for the dataset/loading snippet and the numbered `Tasks` list.
2. If the dataset needs a package not in `requirements.txt` (e.g. `seaborn`), add it there and `pip install` it into `.venv` — the file must actually run.
3. Write `templates/NN_slug.py` per the Output Contract above.
4. Run it: `source .venv/bin/activate && python templates/NN_slug.py`. It must exit 0, print dataset size + sample rows, no traceback. Fix and re-run until clean — not optional.

## Common Mistakes

- Calling the stub functions from `main()` "to show it works" — they're `pass`, so this crashes or silently returns `None`. Keep that wiring commented out.
- Printing only `df.head()` without row/column counts — both are required.
- Mismatched filename (md is `03_x.md`, template becomes `3_x.py` or a different slug) — copy the basename exactly, swap the extension.
- Needing a dataset package that isn't installed yet and skipping the verification run instead of installing it.
