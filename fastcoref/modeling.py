import logging
from abc import ABC
from typing import List

import numpy as np
import pandas as pd
import spacy
import torch
import transformers
from spacy.cli import download
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, BatchEncoding

from fastcoref.coref_models.modeling_fcoref import FCorefModel
from fastcoref.coref_models.modeling_lingmess import LingMessModel
from fastcoref.utilities.collate import LeftOversCollator, DynamicBatchSampler, \
    PadCollator
from fastcoref.utilities.util import set_seed, create_mention_to_antecedent, \
    create_clusters, align_to_char_level, \
    encode

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - \t %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


class CorefResult:
    def __init__(self, text, clusters, char_map, reverse_char_map, coref_logit, text_idx):
        self.text = text
        self.clusters = clusters
        self.char_map = char_map
        self.reverse_char_map = reverse_char_map
        self.coref_logit = coref_logit
        self.text_idx = text_idx

    def get_clusters(self, as_strings=True):
        if not as_strings:
            return [[self.char_map[mention][1] for mention in cluster] for cluster in self.clusters]

        return [[self.text[self.char_map[mention][1][0]:self.char_map[mention][1][1]] for mention in cluster]
                for cluster in self.clusters]

    def get_logit(self, span_i, span_j):
        if span_i not in self.reverse_char_map:
            raise ValueError(f'span_i="{self.text[span_i[0]:span_i[1]]}" is not an entity in this model!')
        if span_j not in self.reverse_char_map:
            raise ValueError(f'span_i="{self.text[span_j[0]:span_j[1]]}" is not an entity in this model!')

        span_i_idx = self.reverse_char_map[span_i][0]   # 0 is to get the span index
        span_j_idx = self.reverse_char_map[span_j][0]

        if span_i_idx < span_j_idx:
            return self.coref_logit[span_j_idx, span_i_idx]

        return self.coref_logit[span_i_idx, span_j_idx]

    def __str__(self):
        if len(self.text) > 50:
            text_to_print = f'{self.text[:50]}...'
        else:
            text_to_print = self.text
        return f'CorefResult(text="{text_to_print}", clusters={self.get_clusters()})'

    def __repr__(self):
        return self.__str__()


class CorefModel(ABC):
    def __init__(self, model_name_or_path, coref_class, collator_class, enable_progress_bar, device=None, nlp=None, max_model_tokens: int = 10000):
        """Initializer.

        Args:
            model_name_or_path: the model name if loading from huggingface hub or the path to a checkpoint file
            enable_progress_bar: whether to enable a progress bar
            device: 'cpu' or 'cuda'
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.seed = 42
        self._set_device()
        self.enable_progress_bar = enable_progress_bar

        config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.max_segment_len = config.coref_head['max_segment_len']
        self.max_doc_len = config.coref_head['max_doc_len'] if 'max_doc_len' in config.coref_head else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True,
            add_prefix_space=True, verbose=False, model_max_length=max_model_tokens
        )

        if collator_class == PadCollator:
            self.collator = PadCollator(tokenizer=self.tokenizer, device=self.device)
        elif collator_class == LeftOversCollator:
            self.collator = LeftOversCollator(
                tokenizer=self.tokenizer, device=self.device,
                max_segment_len=config.coref_head['max_segment_len']
            )
        else:
            raise NotImplementedError(f"Class collator {type(collator_class)} is not supported! "
                                      f"only LeftOversCollator or PadCollator supported")
        if nlp is not None:
            self.nlp = nlp
        else:
            try:
                self.nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
            except OSError:
                # TODO: this is a workaround it is not clear how to add "en_core_web_sm" to setup.py
                download('en_core_web_sm')
                self.nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])

        self.model, loading_info = coref_class.from_pretrained(
            self.model_name_or_path, config=config,
            output_loading_info=True
        )
        self.model.to(self.device)

        set_seed(self)
        transformers.logging.set_verbosity_error()

    def _set_device(self):
        """Sets the device of the model 'cpu' or 'cuda'."""

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)
        self.n_gpu = torch.cuda.device_count()

    def _create_dataset(self, texts: List[str]) -> pd.DataFrame:
        """Compile a pandas dataframe for the model inference.

        Args:
            texts: List of strings representing sentences/paragraphs for inference

        Returns:
            A dataframe dataset
        """

        logger.info("Tokenize %d inputs...", len(texts))

        # Save original text ordering for later use
        dataset = {"text": texts, "idx": range(len(texts))}
        dataset.update(encode(dataset, self.tokenizer, self.nlp))
        dataset = pd.DataFrame.from_dict(dataset)  # pyright: ignore

        return dataset

    def _prepare_batches(
        self, dataset: pd.DataFrame, max_tokens_in_batch: int
    ) -> DynamicBatchSampler:
        """Create dynamic batches.

        Args:
            dataset: a dataframe dataset
            max_tokens_in_batch: maximum number of tokens in a single input batch

        Returns:
            an instance of DynamicBatchSampler as a dataloader
        """

        dataloader = DynamicBatchSampler(
            dataset,
            collator=self.collator,
            max_tokens=max_tokens_in_batch,
            max_segment_len=self.max_segment_len,
            max_doc_len=self.max_doc_len,  # type: ignore
        )

        return dataloader

    # pylint: disable=too-many-locals
    def _batch_inference(self, batch: BatchEncoding) -> List[CorefResult]:
        """Does inference on batches of data.

        Args:
            batch: an encoded batch of input

        Returns:
            List of CorefResult instances
        """

        texts = batch["text"]
        subtoken_map = batch["subtoken_map"]
        token_to_char = batch["offset_mapping"]
        idxs = batch["idx"]
        with torch.no_grad():
            outputs = self.model(batch, return_all_outputs=True)  # pyright: ignore

        outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)

        span_starts, span_ends, _, coref_logits = outputs_np
        doc_indices, mention_to_antecedent = create_mention_to_antecedent(
            span_starts, span_ends, coref_logits
        )

        results = []

        for i, _ in enumerate(texts):  # type: ignore
            doc_mention_to_antecedent = mention_to_antecedent[
                np.nonzero(doc_indices == i)
            ]
            predicted_clusters = create_clusters(doc_mention_to_antecedent)

            char_map, reverse_char_map = align_to_char_level(
                span_starts[i], span_ends[i], token_to_char[i], subtoken_map[i]  # type: ignore
            )

            result = CorefResult(
                text=texts[i],  # type: ignore
                clusters=predicted_clusters,  # type: ignore
                char_map=char_map,
                reverse_char_map=reverse_char_map,
                coref_logit=coref_logits[i],
                text_idx=idxs[i],  # type: ignore
            )

            results.append(result)

        return results

    def _inference(self, dataloader: DynamicBatchSampler) -> List[CorefResult]:
        """Run inference using a dynamic data loader.

        Args:
            dataloader: a dataloader instance

        Returns:
            List of CorefResult instance
        """

        self.model.eval()  # pyright: ignore
        logger.info(
            "***** Running Inference on %d texts *****", len(dataloader.dataset)
        )

        results = []
        if self.enable_progress_bar:
            with tqdm(desc="Inference", total=len(dataloader.dataset)) as progress_bar:
                for batch in dataloader:
                    results.extend(self._batch_inference(batch))  # type: ignore
                    progress_bar.update(n=len(batch["text"]))  # type: ignore
        else:
            for batch in dataloader:
                results.extend(self._batch_inference(batch))  # type: ignore

        return sorted(results, key=lambda res: res.text_idx)

    def predict(
        self,
        texts: List[str],  # similar to huggingface tokenizer inputs
        max_tokens_in_batch: int = 10000,
    ) -> List[CorefResult]:
        """Predict the coref clusters of a list of texts.

        Args:
            texts: The sequence to be encoded. Each sequence should be a string.
            max_tokens_in_batch: maximum number of tokens in a single input batch

        Returns:
            List of CorefResult instance
        """

        # Input type checking for clearer error
        # pylint: disable=no-else-return
        def _is_valid_text_input(texts):
            if isinstance(texts, (list, tuple)):
                # List are fine as long as they are...
                if len(texts) == 0:
                    # ... empty
                    return True
                elif all([isinstance(t, str) for t in texts]):
                    # ... list of strings
                    return True
                else:
                    return False
            else:
                return False

        if not _is_valid_text_input(texts):
            raise ValueError(
                "text input must be of type `List[str]` (batch or single pretokenized example) "
            )

        dataset = self._create_dataset(texts)
        dataloader = self._prepare_batches(dataset, max_tokens_in_batch)

        preds = self._inference(dataloader)
        return preds


class FCoref(CorefModel):
    def __init__(self, model_name_or_path='biu-nlp/f-coref', device=None, nlp=None, enable_progress_bar=True, max_model_tokens: int = 10000):
        super().__init__(model_name_or_path, FCorefModel, LeftOversCollator, enable_progress_bar, device, nlp, max_model_tokens)


class LingMessCoref(CorefModel):
    def __init__(self, model_name_or_path='biu-nlp/lingmess-coref', device=None, nlp=None, enable_progress_bar=True, max_model_tokens: int = 10000):
        super().__init__(model_name_or_path, LingMessModel, PadCollator, enable_progress_bar, device, nlp, max_model_tokens)
