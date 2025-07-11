"""
This file contains a temporary implementation of the custom MathLens dataset for dissecting multimodal reasoning behaviour.
It is "temporary" both in the sense that the benchmark is not final and public.
Hence we use the local path.
"""

import os
from typing import Any, Optional

from eureka_ml_insights.core import (
    DataProcessing,
    EvalReporting,
    Inference,
    PromptProcessing,
)
from eureka_ml_insights.data_utils import (
    AddColumn,
    ColumnRename,
    DataReader,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
)

from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataProcessingConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)

from eureka_ml_insights.data_utils.mathverify_utils import MathVerifyOutputEvaluator
from eureka_ml_insights.metrics.reports import AverageAggregator
from eureka_ml_insights.configs import ExperimentConfig


class MATHLENS_PIPELINE(ExperimentConfig):
    # mathlens_data_path: str = "../mathlens"
    mathlens_data_path: str = os.environ.get(
        "MATHLENS_PATH", "microsoft/mathlens"
    )  # "../mm_reasoning/data/data/ours/geometry/mathlens"

    mathlens_setup_name: str = ""  # default
    mathlens_data_split: str = "test"
    mathlens_question_key: str = "question_vis"
    mathlens_use_images: bool = True
    mathlens_per_key_aggregation: list[tuple[str, str]] = []

    def configure_pipeline(
        self,
        model_config: ModelConfig,
        resume_from: Optional[str] = None,
        **kwargs: dict[str, Any],
    ) -> PipelineConfig:
        mathlens_data_is_local: bool = "MATHLENS_PATH" in os.environ

        # Configure the data processing component.
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": self.mathlens_data_path,
                    "split": self.mathlens_data_split,
                    "transform": SequenceTransform(
                        [
                            ColumnRename(
                                name_mapping={
                                    self.mathlens_question_key: "prompt",
                                    "answer": "ground_truth",
                                }
                            ),
                        ]
                    ),
                    "load_data_from_disk": mathlens_data_is_local,
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__),
                "../prompt_templates/mathlens_templates/question.jinja",
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        _data_config = {
            "path": os.path.join(
                self.data_processing_comp.output_dir, "transformed_data.jsonl"
            )
        }
        if self.mathlens_use_images:
            _data_config["image_column_names"] = ["decoded_image"]
        else:
            _data_config["load_images"] = False

        # Configure the inference component
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(MMDataLoader, _data_config),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
            max_concurrent=10,
        )

        # post process the response to extract the answer
        self.data_post_processing = DataProcessingConfig(
            component_type=DataProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(
                        self.inference_comp.output_dir, "inference_result.jsonl"
                    ),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            AddColumn("score"),
                            MathVerifyOutputEvaluator(score_column_name="score"),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_post_processing_output"),
        )

        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(
                        self.data_post_processing.output_dir, "transformed_data.jsonl"
                    ),
                    "format": ".jsonl",
                },
            ),
            aggregator_configs=[
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": ["score"],
                        "filename_base": f"MathLens{self.mathlens_setup_name}_Score",
                        "per_key_aggregation": [*self.mathlens_per_key_aggregation],
                    },
                ),
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": ["score"],
                        "filename_base": f"MathLens{self.mathlens_setup_name}_Score_AllCorrect",
                        "per_key_aggregation": [
                            *self.mathlens_per_key_aggregation,
                            ("problem_id", "min"),
                        ],
                    },
                ),
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": ["score"],
                        "filename_base": f"MathLens{self.mathlens_setup_name}_Score_AnyCorrect",
                        "per_key_aggregation": [
                            *self.mathlens_per_key_aggregation,
                            ("problem_id", "max"),
                        ],
                    },
                ),
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": ["score"],
                        "filename_base": f"MathLens{self.mathlens_setup_name}_Score_By_Modification",
                        "group_by": ["modification_type"],
                        "per_key_aggregation": [*self.mathlens_per_key_aggregation],
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.data_post_processing,
                self.evalreporting_comp,
            ],
            self.log_dir,
        )


class MATHLENS_TEXT_PIPELINE(MATHLENS_PIPELINE):
    mathlens_setup_name: str = "TEXT"
    mathlens_data_split: str = "test"
    mathlens_question_key: str = "question_text"
    mathlens_use_images: bool = False


class MATHLENS_PERCEPTION_PIPELINE(MATHLENS_PIPELINE):
    mathlens_setup_name: str = "PERCEPTION"
    mathlens_data_split: str = "perception"
    mathlens_question_key: str = "question"
    mathlens_per_key_aggregation: list[tuple[str, str]] = [("image_key", "min")]


class MATHLENS_PERCEPTIONBASE_PIPELINE(MATHLENS_PERCEPTION_PIPELINE):
    mathlens_setup_name: str = "PERCEPTIONBASE"
    mathlens_data_split: str = "perception_base"


class MATHLENS_DEBUG_PIPELINE(MATHLENS_PIPELINE):
    mathlens_setup_name: str = "DEBUG"
    mathlens_data_split: str = "debug"
