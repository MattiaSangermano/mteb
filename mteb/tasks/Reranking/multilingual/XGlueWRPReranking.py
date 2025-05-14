from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

import datasets
import pandas as pd

logger = logging.getLogger(__name__)


_LANGUAGES = {
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "it": ["ita-Latn"],
    "pt": ["por-Latn"],
    "zh": ["zho-Hans"]
}

_CITATION = """
    @article{Liang2020XGLUEAN,
    title={XGLUE: A New Benchmark Dataset for Cross-lingual Pre-training, Understanding and Generation},
    author={Yaobo Liang and Nan Duan and Yeyun Gong and Ning Wu and Fenfei Guo and Weizhen Qi and Ming Gong and Linjun Shou and Daxin Jiang and Guihong Cao and Xiaodong Fan and Ruofei Zhang and Rahul Agrawal and Edward Cui and Sining Wei and Taroon Bharti and Ying Qiao and Jiun-Hung Chen and Winnie Wu and Shuguang Liu and Fan Yang and Daniel Campos and Rangan Majumder and Ming Zhou},
    journal={arXiv},
    year={2020},
    volume={abs/2004.01401}
    }

    @article{Sang2003IntroductionTT,
    title={Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition},
    author={Erik F. Tjong Kim Sang and Fien De Meulder},
    journal={ArXiv},
    year={2003},
    volume={cs.CL/0306050}
    }

    @article{Sang2002IntroductionTT,
    title={Introduction to the CoNLL-2002 Shared Task: Language-Independent Named Entity Recognition},
    author={Erik F. Tjong Kim Sang},
    journal={ArXiv},
    year={2002},
    volume={cs.CL/0209010}
    }

    @inproceedings{Conneau2018XNLIEC,
    title={XNLI: Evaluating Cross-lingual Sentence Representations},
    author={Alexis Conneau and Guillaume Lample and Ruty Rinott and Adina Williams and Samuel R. Bowman and Holger Schwenk and Veselin Stoyanov},
    booktitle={EMNLP},
    year={2018}
    }

    @article{Lewis2019MLQAEC,
    title={MLQA: Evaluating Cross-lingual Extractive Question Answering},
    author={Patrick Lewis and Barlas Oguz and Ruty Rinott and Sebastian Riedel and Holger Schwenk},
    journal={ArXiv},
    year={2019},
    volume={abs/1910.07475}
    }

    @article{Yang2019PAWSXAC,
    title={PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification},
    author={Yinfei Yang and Yuan Zhang and Chris Tar and Jason Baldridge},
    journal={ArXiv},
    year={2019},
    volume={abs/1908.11828}
    }"""


class XGlueWRPReranking(MultilingualTask, AbsTaskReranking):
    metadata = TaskMetadata(
        name="XGlueWRPReranking",
        description="""XGLUE is a new benchmark dataset to evaluate the performance of cross-lingual pre-trained models 
        with respect to cross-lingual natural language understanding and generation. XGLUE is composed of 11 tasks spans 19 languages.""",
        dataset={
            "path": "forresty/xglue",
            "revision": "833b866f2f71a28d7251569020f0ff82ee5fdbbb",
            "name": "wpr"
        },
        type="Reranking",
        category="s2p",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=_LANGUAGES,
        main_score="map",
        domains=["Written"],
        task_subtypes=[],
        license="http://hdl.handle.net/11234/1-3105",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION,
    )

    
    
    def load_data(self, **kwargs):
        def _aggregate_texts(group, neg_label):
            return pd.Series({
                'positive': group[group['relavance_label'] != neg_label]['text'].tolist(),
                'negative': group[group['relavance_label'] == neg_label]['text'].tolist()
            })

        self.dataset = {}
        for lang in self.hf_subsets:
            ds = {}
            for eval_split in self.metadata.eval_splits:

                ds[eval_split] = datasets.load_dataset(
                    split=f"{eval_split}.{lang}",
                    **self.metadata.dataset
                ).map(
                    lambda x: {'text': x['web_page_title'] + x['web_page_snippet']}
                )

                neg_label = ds[eval_split].features['relavance_label']._str2int['Bad']

                grouped_df = ds[eval_split].to_pandas(
                    ).groupby('query'
                            ).apply(
                                _aggregate_texts,
                                neg_label=neg_label
                                ).reset_index()
                
                ds[eval_split] = datasets.Dataset.from_pandas(grouped_df)

            self.dataset[lang] = datasets.DatasetDict(ds)
