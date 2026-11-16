# Graph Report - witches_brew  (2026-11-16)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 645 nodes · 1579 edges · 32 communities (20 shown, 12 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 79 edges (avg confidence: 0.89)
- Token cost: 36,123 input · 464 output

## Graph Freshness
- Built from commit: `b1fee957`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Low-Frequency Purification Defense
- Result Aggregation CLI
- Dataset Loading and Cases
- Poison Brewing Loop
- Experiment Config Loading
- Training Diagnostics
- Perturbation Spectrum Analysis
- End-to-End Pipeline Tests
- Victim Model Training
- Case Evaluation and Surrogates
- Brew CLI and Export
- Feature Outlier Filtering
- Attack Component Unit Tests
- CIFAR ResNet Models
- Experiment Runner and Results
- Model Factory
- Poisoning Objectives
- Seeding and Timing Utils
- Differentiable Random Transform
- Target Loss Functions
- Six-Layer ConvNet
- Remote Bootstrap Script
- Target Listing Tool
- Witches Brew Package

## God Nodes (most connected - your core abstractions)
1. `load_config()` - 29 edges
2. `train_model()` - 27 edges
3. `run_case()` - 26 edges
4. `build_case()` - 25 edges
5. `main()` - 24 edges
6. `Brewer` - 22 edges
7. `run_filtering()` - 21 edges
8. `main()` - 20 edges
9. `config_from_args()` - 19 edges
10. `main()` - 19 edges

## Surprising Connections (you probably didn't know these)
- `main()` --uses--> `Timer`  [INFERRED]
  scripts/brew_case.py → src/witchesbrew/utils.py
- `main()` --uses--> `Timer`  [INFERRED]
  scripts/poison_custom.py → src/witchesbrew/utils.py
- `test_flip_is_a_horizontal_mirror()` --uses--> `RandomTransform`  [INFERRED]
  tests/test_components.py → src/witchesbrew/data/augment.py
- `fake_data()` --uses--> `IndexedDataset`  [INFERRED]
  tests/test_pipeline.py → src/witchesbrew/data/datasets.py
- `main()` --uses--> `Brewer`  [INFERRED]
  scripts/brew_case.py → src/witchesbrew/attacks/brew.py

## Import Cycles
- None detected.

## Communities (32 total, 12 thin omitted)

### Community 0 - "Low-Frequency Purification Defense"
Cohesion: 0.06
Nodes (43): lowpass_normalized(), subspace_project(), build_augmentation(), _basis_for(), _blockify(), brew_purifier_normalized(), build_purifier(), dct2() (+35 more)

### Community 1 - "Result Aggregation CLI"
Cohesion: 0.07
Nodes (24): collect(), main(), announce(), build_parser(), config_from_args(), _load_case(), main(), plot_alignment() (+16 more)

### Community 2 - "Dataset Loading and Cases"
Cohesion: 0.06
Nodes (27): compute_mean_std(), image_size(), IndexedDataset, normalization(), _require_custom(), _resolve_custom_stats(), build_case(), _class_index() (+19 more)

### Community 3 - "Poison Brewing Loop"
Cohesion: 0.06
Nodes (23): Brewer, initialize_delta(), PoisonBounds, _scheduler_milestones(), current_objective(), double_model(), small_problem(), test_brewing_reduces_the_alignment_loss() (+15 more)

### Community 4 - "Experiment Config Loading"
Cohesion: 0.06
Nodes (20): _coerce(), Config, load_config(), _rederive(), variants(), dp_sweep_configs(), make_gradient_hook(), test_base_config_matches_the_published_hyperparameters() (+12 more)

### Community 5 - "Training Diagnostics"
Cohesion: 0.09
Nodes (14): adversarial_gradient(), epoch_alignment(), _flat_grad(), make_diagnostics_fn(), diagnostics(), OneBatch, plain_model(), test_alignment_equals_one_minus_the_brewing_objective() (+6 more)

### Community 6 - "Perturbation Spectrum Analysis"
Cohesion: 0.11
Nodes (15): main(), alignment_survival(), cos_to_target(), budget_survival(), delta_report(), delta_to_pixel(), offmanifold_energy(), _poison_gradient() (+7 more)

### Community 7 - "End-to-End Pipeline Tests"
Cohesion: 0.09
Nodes (14): fake_data(), FakeCIFAR, test_data_ablation_shrinks_the_victim_training_set(), __init__(), test_perturbation_reaches_the_victim_by_dataset_index(), __init__(), test_poisons_are_drawn_from_the_adversarial_class(), test_run_case_produces_the_expected_record() (+6 more)

### Community 8 - "Victim Model Training"
Cohesion: 0.11
Nodes (11): make_loader(), ablated_subset(), apply_privacy_defense(), build_optimizer(), check_targets(), evaluate_accuracy(), inject_poison(), _poison_lookup() (+3 more)

### Community 9 - "Case Evaluation and Surrogates"
Cohesion: 0.15
Nodes (13): run_filtering(), augmentation_for(), build_datasets(), num_classes(), build_surrogate_ensemble(), clean_baseline_accuracy(), get_surrogate(), run_case() (+5 more)

### Community 10 - "Brew CLI and Export"
Cohesion: 0.18
Nodes (9): main(), main(), export_imagefolder(), export_packed(), export_png(), load_packed(), quantization_error(), quantize() (+1 more)

### Community 11 - "Feature Outlier Filtering"
Cohesion: 0.13
Nodes (5): extract_features(), feature_statistics(), filter_outliers(), FilterReport, random_guessing_baseline()

### Community 12 - "Attack Component Unit Tests"
Cohesion: 0.14
Nodes (10): _cosine_global(), test_convnet_penultimate_width_matches_the_paper(), test_cosine_objective_is_bounded(), test_cosine_objective_is_two_on_opposed_gradients(), test_cosine_objective_vanishes_on_aligned_gradients(), test_eps_bound_is_in_pixel_units(), test_euclidean_objective_vanishes_only_on_equality(), test_flip_is_a_horizontal_mirror() (+2 more)

### Community 13 - "CIFAR ResNet Models"
Cohesion: 0.17
Nodes (6): BasicBlock, ResNet, resnet18(), resnet34(), test_cifar_resnet_stem_has_no_downsampling(), test_resnet_width_argument_scales_the_network()

### Community 14 - "Experiment Runner and Results"
Cohesion: 0.21
Nodes (4): run_experiment(), mean_and_stderr(), ResultStore, test_standard_error_matches_the_paper_definition()

### Community 15 - "Model Factory"
Cohesion: 0.21
Nodes (5): bypass_last_layer(), _cifar_mobilenet_v2(), _cifar_vgg(), feature_dim(), _PoolAndFlatten

### Community 16 - "Poisoning Objectives"
Cohesion: 0.23
Nodes (5): bullseye_loss(), _cosine_per_layer(), _euclidean(), gradient_matching_loss(), poison_frogs_loss()

### Community 17 - "Seeding and Timing Utils"
Cohesion: 0.22
Nodes (3): available_workers(), set_seed(), Timer

### Community 19 - "Target Loss Functions"
Cohesion: 0.39
Nodes (4): carlini_wagner(), cross_entropy(), get_target_criterion(), test_carlini_wagner_is_negative_when_the_intended_class_wins()

### Community 21 - "Remote Bootstrap Script"
Cohesion: 0.40
Nodes (4): PIP_CACHE_DIR, bootstrap_remote.sh script, TORCH_HOME, XDG_CACHE_HOME

## Knowledge Gaps
- **4 isolated node(s):** `PIP_CACHE_DIR`, `TORCH_HOME`, `XDG_CACHE_HOME`, `witchesbrew`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 257 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_config()` connect `Experiment Config Loading` to `Low-Frequency Purification Defense`, `Result Aggregation CLI`, `Dataset Loading and Cases`, `Poison Brewing Loop`, `End-to-End Pipeline Tests`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `run_case()` (e.g. with `.brew()` and `PoisonCase`) actually correct?**
  _`run_case()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `PIP_CACHE_DIR`, `TORCH_HOME`, `XDG_CACHE_HOME` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Low-Frequency Purification Defense` be split into smaller, more focused modules?**
  _Cohesion score 0.05527805527805528 - nodes in this community are weakly interconnected._
- **Why does `run_filtering()` connect `Case Evaluation and Surrogates` to `Low-Frequency Purification Defense`, `Result Aggregation CLI`, `Dataset Loading and Cases`, `Poison Brewing Loop`, `Victim Model Training`, `Brew CLI and Export`, `Feature Outlier Filtering`, `Seeding and Timing Utils`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Should `Result Aggregation CLI` be split into smaller, more focused modules?**
  _Cohesion score 0.07462686567164178 - nodes in this community are weakly interconnected._
- **Why does `train_model()` connect `Victim Model Training` to `Low-Frequency Purification Defense`, `Perturbation Spectrum Analysis`, `End-to-End Pipeline Tests`, `Case Evaluation and Surrogates`, `Brew CLI and Export`, `Seeding and Timing Utils`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._