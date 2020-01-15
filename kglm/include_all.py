from allennlp.models.archival import load_archive
from allennlp.models import Model

import json

from kglm.commands.complete_the_sentence import CompleteTheSentence
from kglm.commands.evaluate_perplexity import EvaluatePerplexity
from kglm.commands.evaluate_perplexity import evaluate_perplexity, evaluate_from_args
from kglm.commands.sample import Sample
from kglm.data.dataset_readers.enhanced_wikitext import EnhancedWikitextKglmReader, \
    EnhancedWikitextSimpleKglmReader, EnhancedWikitextReader, EnhancedWikitextEntityNlmReader
from kglm.data.fields.global_object import GlobalObject
from kglm.data.fields.sequential_array import SequentialArrayField
from kglm.data.iterators.fancy_iterator import FancyIterator,
from kglm.data.alias_database import AliasDatabase
from kglm.data.extended_vocabulary import ExtendedVocabulary
from kglm.models.kglm import Kglm
from kglm.models.kglm_disc import KglmDisc
from kglm.models.awd_lstm import AwdLstmLanguageModel
from kglm.modules.dynamic_embeddings import DynamicEmbedding
from kglm.modules.embed_regularize import embedded_dropout
from kglm.modules.knowledge_graph_lookup import KnowledgeGraphLookup
from kglm.modules.locked_dropout import LockedDropout
from kglm.modules.recent_entities import RecentEntities
from kglm.modules.splitcross import SplitCrossEntropyLoss
from kglm.modules.weight_drop import WeightDrop
from kglm.nn.util import *
from kglm.predictors.kglm import KglmPredictor
from kglm.predictors.complete_the_sentence import CompleteTheSentencePredictor

@Model.register("from_archive")
class ModelArchiveFromParams(Model):
    """
    Loads a model from an archive
    """
    @classmethod
    def from_params(cls, vocab=None, params=None):
        """
        {"type": "from_archive", "archive_file": "path to archive",
         "overrides:" .... }

        "overrides" omits the "model" key
        """
        archive_file = params.pop("archive_file")
        overrides = params.pop("overrides", None)
        params.assert_empty("ModelArchiveFromParams")
        if overrides is not None:
            archive = load_archive(archive_file, overrides=json.dumps(overrides))
        else:
            archive = load_archive(archive_file)
        return archive.model