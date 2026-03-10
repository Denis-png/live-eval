# Team Project on topic of Evaluation Data Generation on the fly

## Pipeline

1. **G**enerate evaluation data based on existing datasets.
2. **E**valuate the task model using generated(synthetic) data.
3. **T**rash generated data to prevent leakages to training data.

## Parameters

1. Evaluation(test) dataset
2. LLM/Non-LLM for generation
3. Task-dependent model to evaluate
4. Metric/benchmark for evaluation

## General plan

1. Implement live evaluation on Grammatical Error Correction(GEC) models
. . .
Final. Deploy final python framework for evaluation of different task models on the fly
