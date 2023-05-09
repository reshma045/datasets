# coding=utf-8
# Copyright 2023 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""bot_adversarial_dialogue dataset."""
import os
from typing import Any, List, Mapping, Tuple

from etils import epath
import numpy as np

from tensorflow_datasets.core.utils import bool_utils
import tensorflow_datasets.public_api as tfds


_BOT_ADVERSARIAL_DIALOGUE_DATASETS_VERSION = "v0.2"
_HUMAN_NONADV_SAFETY_EVAL_TESTSET_VERSION = "v0.1"

# Class labels in "dialogue_datasets" and "human_nonadv_safety_eval" configs.
_LABELS = tfds.features.ClassLabel(names=["__ok__", "__notok__"])

# Features which are common to all configs.
_COMMON_FEATURES = {
    "id": tfds.features.Text(),
    "text": tfds.features.Text(),
    "episode_done": np.bool_,
    "labels": _LABELS,
}

# Config-specific features.
_DIALOGUE_FEATURES = {
    "speaker_to_eval": tfds.features.Text(),
    "human_persona": tfds.features.Sequence(tfds.features.Text()),
    "previous_dialogue_acts": tfds.features.Sequence(tfds.features.Text()),
}
_CONFIG_FEATURES = {
    "dialogue_datasets": tfds.features.FeaturesDict(
        {**_DIALOGUE_FEATURES, **_COMMON_FEATURES}
    ),
    "human_nonadv_safety_eval": tfds.features.FeaturesDict(_COMMON_FEATURES),
}


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for bot_adversarial_dialogue dataset."""

  VERSION = tfds.core.Version("1.0.0")
  RELEASE_NOTES = {
      "1.0.0": "Initial release.",
  }
  BUILDER_CONFIGS = [
      tfds.core.BuilderConfig(
          name="dialogue_datasets",
          description=(
              "The dialogue datasets, divided in train, validation and test"
              " splits."
          ),
      ),
      tfds.core.BuilderConfig(
          name="human_nonadv_safety_eval",
          description=(
              "An human safety evaluation set evaluated by crowdsourced workers"
              " for offensiveness. "
          ),
      ),
  ]
  DEFAULT_CONFIG_NAME = "dialogue_datasets"

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=_CONFIG_FEATURES[self.builder_config.name],
        supervised_keys=None,
        homepage="https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/bot_adversarial_dialogue",
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    bot_adversarial_dialogue_home = (
        "http://parl.ai/downloads/bot_adversarial_dialogue/"
    )

    if self.builder_config.name == "dialogue_datasets":
      path = dl_manager.download_and_extract(
          os.path.join(
              bot_adversarial_dialogue_home,
              f"dialogue_datasets_{_BOT_ADVERSARIAL_DIALOGUE_DATASETS_VERSION}.tar.gz",
          )
      )

      return {
          "train": self._generate_examples(
              path / "bot_adversarial_dialogue_datasets_with_persona/train.txt",
              split_name="train",
          ),
          "valid": self._generate_examples(
              path / "bot_adversarial_dialogue_datasets_with_persona/valid.txt",
              split_name="valid",
          ),
          "test": self._generate_examples(
              path / "bot_adversarial_dialogue_datasets_with_persona/test.txt",
              split_name="test",
          ),
      }

    else:
      path = dl_manager.download_and_extract(
          os.path.join(
              bot_adversarial_dialogue_home,
              f"human_nonadv_safety_eval_{_HUMAN_NONADV_SAFETY_EVAL_TESTSET_VERSION}.tar.gz",
          )
      )

      return {
          "test": self._generate_examples(
              path / "human_nonadv_safety_eval/test.txt",
              split_name="human_nonadv_safety_eval",
          ),
      }

  def _generate_examples(self, path, split_name=str):
    """Yields examples."""

    def _preprocess_row(row: str) -> str:
      """Preprocesses a dataset row using ParlAI format.

      This function is based on:
      https://github.com/facebookresearch/ParlAI/blob/9974b947fb2e801dc5608f495828532c2a714742/parlai/utils/misc.py#L639

      Args:
        row: An unprocessed row from the bot_adversarial_dialogue dataset.

      Returns:
        A processed row, in which special characters are properly formatted.
      """
      row = str(row)
      row = row.replace("\\t", "\t")
      row = row.replace("\\n", "\n")
      row = row.replace("__PIPE__", "|")
      return row

    def _get_row_features(row: str) -> Mapping[str, Any]:
      """Extracts dialogue features from a dataset row."""
      row_features = {}
      for field in row.split("\t"):
        idx = field.find(":")
        key, value = field[:idx], field[idx + 1 :]
        row_features[key] = value
      return row_features

    def _get_text_and_dialogue_acts(row: str) -> Tuple[str, List[str]]:
      """Extracts the previous dialogue acts from the text."""
      if "\n" in row:
        acts = row.split("\n")
        return acts[-1], acts[:-1]
      # The first episode of a dialogue doesn't have any previous dialogue acts.
      else:
        return row, []

    with epath.Path(path).open() as f:
      for i, row in enumerate(f):
        example_id = f"{split_name}_{i}"
        cleaned_row = _preprocess_row(row)
        row_features = _get_row_features(cleaned_row)

        example = {
            "id": row_features.get("id", example_id),
            "labels": row_features["labels"],
            "episode_done": bool_utils.parse_bool(row_features["episode_done"]),
        }

        if self.builder_config.name == "dialogue_datasets":
          text, previous_dialogue_acts = _get_text_and_dialogue_acts(
              row_features["text"]
          )
          human_persona = [
              str_.strip()
              for str_ in row_features["bot_persona"].strip().split("\n")
          ]

          example.update({
              "text": text,
              "previous_dialogue_acts": previous_dialogue_acts,
              "human_persona": human_persona,
              "speaker_to_eval": row_features["speaker_to_eval"],
          })
        else:
          example["text"] = row_features["text"]

        yield example_id, example
