from abc import ABC, abstractmethod


class BaseTask(ABC):

    @abstractmethod
    def get_error_types(self) -> list[str]:
        """
        List of corruption types the generator LLM will introduce.

        GEC example:   ["subject-verb disagreement", "wrong tense"]
        Spam example:  ["add urgency phrase", "add fake offer"]
        """
        pass

    @abstractmethod
    def get_prompt_instruction(self) -> str:
        """
        Prompt template for the generator LLM.
        Must contain a {sentence} placeholder; may optionally use {error_type}
        (the generator always passes both to str.format).
        """
        pass

    @abstractmethod
    def get_evaluators(self) -> list[str]:
        """
        Evaluator names to compute for this task.
        Must match keys returned by get_evaluator_fns().
        """
        pass

    @abstractmethod
    def get_evaluator_fns(self) -> dict:
        """
        Map of evaluator name → callable(results: list[dict]) → score.
        results dicts contain: original, corrupted, prediction.
        """
        pass

    @abstractmethod
    def get_model(self, model_config: dict):
        """
        Return a BaseModel instance for the given model config.
        The task owns the mapping from model type to model class.
        """
        pass

    def get_judge_prompt(self) -> str | None:
        """
        Optional LLM-as-judge prompt for filtering bad generations.
        Must contain {sentence} and {correction} placeholders.
        Return None to disable the judge step.
        """
        return None

    def get_inverse_prompt(self) -> str | None:
        """Inverse-mode prompt template. Must contain {sentence} (the clean source
        text) and {error_spec} (human description of the errors to inject).
        Return None if the task does not support inverse generation."""
        return None

    def get_inverse_judge_prompt(self) -> str | None:
        """Optional inverse-mode judge template with {sentence} and {correction}
        placeholders. Return None to disable judging in inverse mode."""
        return None

    def get_error_descriptions(self) -> dict[str, str]:
        """Map of corruption category key -> short human phrase, used to render
        {error_spec} for the inverse prompt. The keys also define the category
        vocabulary the empirical error distribution samples over.
        Return {} if the task does not support inverse generation."""
        return {}

    def get_carrier_prompt(self) -> str | None:
        """Prompt that synthesizes ONE seed text from a profile content spec.
        Template placeholder: {spec}. Return None if the task has no seedless
        support."""
        return None

    def get_seedless_forward_prompt(self) -> str | None:
        """Forward-mode prompt driven by a content spec instead of a real seed.
        Placeholders: {spec}, {error_spec}. Return None if unsupported."""
        return None

    def get_forward_prompts(self) -> dict[str, str]:
        """{label: prompt} for same-class imitation in classification forward
        mode. Placeholder {sentence}; the positive class also gets {error_spec}
        so forward generation can target the empirical signal mix instead of
        inheriting whatever signals its seed happened to carry."""
        return {}

    def get_class_labels(self) -> tuple[str, str] | None:
        """(positive, negative) label names for `class_conditional` generation.

        The strategy is label -> text, so the dispatcher needs to know what the
        two classes are CALLED. Returning None means this task is not
        class-conditional; a class_conditional task that returns None fails fast
        rather than being generated under someone else's vocabulary.
        """
        return None

    def get_negative_generation_prompt(self) -> str | None:
        """Prompt that produces an example of the NEGATIVE class from a seed.

        Placeholder {sentence}. The positive class is produced by
        get_inverse_prompt (inject) or get_forward_prompts (imitate); this is
        its counterpart, and every class_conditional task needs it.
        """
        return None

    def get_seedless_class_prompts(self) -> dict[str, str]:
        """{label: prompt} for direct per-class seedless generation.
        Placeholders: {spec} and, for the positive class, {error_spec}."""
        return {}

    def get_profile_side(self, mode: str) -> str:
        """Which profile sub-block ("incorrect"/"correct") seedless generation
        should sample content from for this mode. Ignored by tasks whose
        profiles are keyed per label."""
        return "correct"

    def get_seed_pool(self, config: dict, real_data: list[dict], mode: str,
                      *, seed_weights: dict | None = None, rng=None) -> list[dict]:
        """Rows used as generation seeds. Defaults to the parsed real data;
        override when a mode needs a differently-shaped pool (e.g. labeled
        rows of both classes for classification forward mode).

        `seed_weights` is the calibrated per-edit-type weighting for cells whose
        only control input is seed choice (GEC forward+seeded); tasks that do
        not use it ignore both it and `rng`."""
        return real_data

    def get_generation_strategy(self) -> str:
        """How the pipeline generates synthetic data for this task:
          "corruption"        — corrupt a source text (forward/inverse); text→text tasks.
          "class_conditional" — sample a target class, then generate an example of it;
                                classification tasks. Ignores generation.mode.
          "structured"        — generate a whole structured benchmark artifact from
                                a profile/spec; mode is not applicable.
        Default "corruption"."""
        return "corruption"

    def build_structured_generation_prompt(
        self, profile: dict, rng=None, feedback: dict | None = None
    ) -> str:
        """Return one profile-driven structured generation prompt.

        Structured tasks override this. Provider classes remain ontology-agnostic:
        they only receive the prompt string and return text.
        """
        raise NotImplementedError(
            f"{self.get_task_name()} does not support structured generation."
        )

    def parse_structured_generation(self, text: str) -> dict | None:
        """Parse and validate one structured generator response.

        Return None when the model output is malformed or invalid and should be
        skipped rather than entering evaluation.
        """
        raise NotImplementedError(
            f"{self.get_task_name()} does not support structured generation."
        )

    def get_feedback_config(self, generation_config: dict | None = None) -> dict:
        """Per-sample feedback settings for `structured` generation:
        {"enabled": bool, "max_rounds": int, "tolerances": {...}}.

        A structured artifact is a whole sample with its own measurable shape,
        so it can be compared against the reference and regenerated on its own —
        unlike a corruption/classification sample, where fidelity only means
        something across a distribution. Disabled by default.
        """
        return {"enabled": False, "max_rounds": 0}

    def parse_structured_generation_with_diagnostics(self, text: str) -> dict:
        """Parse one structured response into {"artifact", "diagnostic"}.

        Defaults to wrapping `parse_structured_generation`, so a task only has
        to implement the plain parser and the dispatcher needs no branch.
        Override to explain WHY an artifact was rejected.
        """
        artifact = self.parse_structured_generation(text)
        diagnostic: dict = {"valid": artifact is not None}
        if artifact is None:
            diagnostic["rejection_reason"] = "invalid_structured_artifact"
        return {"artifact": artifact, "diagnostic": diagnostic}

    def build_structural_feedback(self, profile: dict, artifact: dict,
                                  generation_config: dict | None = None) -> dict:
        """Compare one generated artifact against the reference profile and
        return {"feedback", "comparison", "synthetic_profile"}.

        Required only when get_feedback_config() enables the loop; the pipeline
        checks that up front and refuses to start rather than failing mid-round.
        """
        raise NotImplementedError(
            f"{self.get_task_name()} enables the structured feedback loop but "
            "does not implement build_structural_feedback()."
        )

    def profile_dataset(self, rows: list[dict]) -> dict | None:
        """Profile a labeled dataset (rows with "text"+"label") for real-vs-generated
        fidelity. Return None to opt out (default). Override in classification tasks."""
        return None

    def compare_profiles(self, real: dict, generated: dict) -> dict | None:
        """Fidelity comparison between two profile_dataset() outputs. Default None."""
        return None

    def get_calibration_keys(self) -> dict[str, str] | None:
        """Map control-input name -> the profile_dataset() key that measures it.

        Lets the calibrator stay as task-agnostic as _sample_categories is: it
        never learns what a "signal" or an "edit type" means, only which key of
        this task's profile measures the distribution it is steering. The two
        tasks name the same concept differently (error_type_dist vs
        signal_type_dist), and this is what reconciles them.

        Return None (default) to opt out of calibration entirely.
        """
        return None

    def get_real_eval_samples(self, config: dict, real_data: list[dict]) -> list[dict] | None:
        """Eval-ready rows for the REAL benchmark, carrying the same schema the
        evaluators expect (classification: text+label; text→text: text+corrupted+
        original). Feeds both the real baseline and real-side profiling. Default
        None → real baseline skipped."""
        return None

    def profile_error_distribution(self, real_data: list[dict],
                                   count_max: int = 5, config: dict | None = None) -> dict | None:
        """Empirical inverse-mode error distribution derived from real_data (and
        optionally the run `config`, e.g. to load an auxiliary subset), keyed on
        get_error_descriptions() vocabulary. Return None when real_data is
        insufficient; the pipeline fails fast on None (default: no empirical
        profiler)."""
        return None

    @abstractmethod
    def get_task_name(self) -> str:
        """
        Short lowercase task identifier, matching configs/<name>/<name>.json and
        data/benchmarks/<name>/ (e.g. "gec"). Must be overridden
        — do NOT derive from the class name, which would drift from the config.
        """
        pass

    def get_label(self, result: dict) -> str | None:
        """
        Return the ground-truth label for a result dict.
        Override in classification tasks to supply the correct label.
        Return None for tasks that don't require a label (e.g. GEC).
        """
        return None

    def get_eval_samples(self, synthetic: list[dict]) -> list[dict]:
        """Expand generated items into rows to classify/score. Each row carries a
        "text" field (the model input). Default: one row per item scoring the
        corrupted text, with the ground-truth "label" from get_label when present.
        Classification tasks may override to add negatives (see SpamTask)."""
        out = []
        for item in synthetic:
            sample = {**item, "text": item["corrupted"]}
            label = self.get_label(sample)
            if label is not None:
                sample["label"] = label
            out.append(sample)
        return out

    @abstractmethod
    def parse_row(self, row: dict) -> dict | None:
        """
        Parse a single raw dataset row into a sample dict.
        Return None to skip the row (e.g. wrong label, missing fields).
        The pipeline collects non-None results up to sample_size.

        GEC example:  maps to {"incorrect": ..., "correct": ...}
        Spam example: filters HAM rows, maps to {"incorrect": ...}
        """
        pass
