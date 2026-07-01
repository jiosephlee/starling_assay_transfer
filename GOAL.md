# Goal of this project

The goal of this project is to create a tool that receives as input

(str) Assay/Phenotype/Property Description
(SMILES) Query Molecule
(SMILES) Retrieved Molecule
(float) (optional) Assay Value of the Retrieved Molecule

and we're trying to predict 

(float) Assay Value of the Query Molecule.

Towards this, we're using /data1/joseph/starling_assay_transfer/datasets/base/Oral_bioavailability_cleaned_v2_tdc_val_test_excluded; 
EDIT: we'll be rebuilding this such that we exclude all the train,val and test molecules of only the Oral Bioavailability task in TDC; previously we took out the val and test molecules across many tasks. We'll call this v3. We'll have to regenerate our pairs, splits, etc. as a result.

as our base dataset that has molecules with the property "oral bioavailability" and generating pairs to train a assay transfer tool.

## Dataset Generation

Our dataset construction pipeline is as follows
1) Generate Pairs
- For each pair, we need to decide what should count as "transfer", "not transfer", and should be dropped to reduce noise in the labels
- For each pair, we need to decide if it's even appropriate to create that pair; for instance, trying to predict an assay used on bats might be just impossible to predict the assay value for humans; this will encourage the model to likely learn spurious correlations; 
    - For our experiments, we have three types of pairing: (a) no constraints (b) same species i.e. same_species_v2 (c) exact match across several columns i.e. "tianang" mode
2) Generate Splits
- Given a list of pairs, we then want to generate the splits for ML development
- For val & test, we want to generate three subsets within val and track them separately
    - 10K where both the query and retrieved are unseen
    - 10K where only the query are unseen
    - 10K were both are unseen
- Train should get the rest of the pairs
- Constraints
    - Zero molecule overlap between splits
    - Val & Test is stratified by label, similarity bucket, missingness across the metadata columns that will be used for the MLP (binary)
3) Generate Prompts
- Then we take each sample and template it via jinja; these are for our LLM experiments

## Function Class

The choice of function class that we're experimenting with are mainly MLPs. However our pipeline should also generate templated prompts that we'll train LLMs with in another codebase; Our repo will simply produce the parquets and upload it to HF.

## Wandb Tracking

HP Sweeps will be tracked in the project "oral_bioavailability_transfer_hp_sweep"

Full runs will be tracked in the project "oral_bioavailability_transfer"

metrics tracked will be consistent across both wandb projects.
- eval/val_{subset}_{metric}
where {subset} is one of the tree val subsets and {metric} is transfer_precision, macro f-1 and accuracy in addition to the default HF training metrics.

## ML Development Pipeline

For each condition, an HP sweep will be done across LRs {1e-4, 2e-4, 4e-4} and effective batch size {8192, 16384, 32768}.
Eval will be done every 50 steps over 300 training steps and the best HP will be determined by macro-f-1 on the "double unseen" subset of val.

Then, during the full training run, we'll keep track of one more additional metric. Every eval we'll also grab the model and do a downstream evaluation on /data1/joseph/starling_assay_transfer/tdc/official_tianang/train/Bioavailability_Ma.jsonl 

We'll take our oral_bioavailabilty_cleaned_v3 dataset and predict the "train" split of TDC regarding oral bioavailability using knn; instead of using morgan fingerprint similarity, however, we'll use the MLP model; for each molecule, we'll take the top 25% based on morgan fingerprint (the weighted version that we have in /data1/joseph/therapeutic-tuning/experiments/knn/oral_bioavailability_transfer) and then take k=10 using the assay_transfer_tool and then take their averaged vote. The cleaned_v3 dataset has oral bioavailability as a percentage so we need to take TDC's rule where below 20% is not orally bioavailable when predicting on TDC. 

This could be expensive so we'll do this every 250 training steps whereas the other metrics are tracked every 50 training steps.

Then we'll save two checkpoints for each condition at the best val (one for macro f-1, and one for best knn downstream macro f-1 on TDC train). Then we'll take the best val checkpoint and evaluate once on the assay transfer test set and once on TDC oral bioavailability val.


