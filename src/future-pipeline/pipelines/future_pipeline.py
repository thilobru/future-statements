import os
import abc
from pathlib import Path
from collections import Counter
import json
from datetime import datetime

import nltk
import numpy as np
import tensorflow as tf
import wandb
from htmldate import find_date
from fastwarc.warc import ArchiveIterator
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse import detect_encoding
from resiliparse.parse.html import HTMLTree
from resiliparse.parse.lang import detect_fast

from helpers import create_s3_client, get_file_stream
from pipelines.prediction_pipeline import PredictionPipeline


class FuturePipeline(PredictionPipeline, abc.ABC):
    """
    """

    def __init__(self, out_file, max_content_length, min_sentence_length, max_sentence_length):
        self.out_file = out_file
        self.max_content_length = max_content_length
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length

        super().__init__()

    def get_signature(self):
        return (
            self.get_tokens_spec(),  # text for classification
            tf.TensorSpec(shape=(), dtype=tf.string),  # url
            tf.TensorSpec(shape=(), dtype=tf.string),  # text for export
            tf.TensorSpec(shape=(), dtype=tf.string),  # timestamp from website
        )

    def get_distributed_filter(self):
        """
        Overridable method that provides a filter, which is executed on the pyspark cluster nodes.
        The returned distributed_filter must not use self. Needed attributes of self should be extracted into variables
        outside of the definition of distributed_filter, which may then use these variables.
        """

        min_length = self.min_sentence_length
        max_length = self.max_sentence_length

        def distributed_filter(text):

            if len(text) < min_length:
                return False

            if len(text) > max_length:
                return False

            if not detect_fast(text)[0] == "en":
                return False

            return True

        return distributed_filter

    def get_tokens_spec(self):
        """
        Overridable method that returns a tf.TensorSpec which corresponds to the values returned by the tokenizer
        defined in get_tokenizer().
        """

        return tf.TensorSpec(shape=(), dtype=tf.string)

    def get_tokenizer(self):
        """
        Overridable method that provides a tokenizer, which is executed on the pyspark cluster nodes.
        The returned tokenizer must not use self. Needed attributes of self should be extracted into variables
        outside of the definition of tokenizer, which may then use these variables.
        """

        def tokenizer(text):
            return text

        return tokenizer

    def get_timestamp_extractor(self):

        def timestamp_extractor(html_tree):
            # return find_date(repr(html_tree.document.html))
            # dummy timestamp:
            return None

        return timestamp_extractor

    def get_generator_factory(self):
        """
        TODO: replace dummy timestamp with actual timestamp from website
        """
        acc_counter = self.acc_counter
        max_content_length = self.max_content_length
        distributed_filter = self.get_distributed_filter()
        tokenizer = self.get_tokenizer()
        timestamp_extractor = self.get_timestamp_extractor()
        AWS_ACCESS_KEY_ID = self.AWS_ACCESS_KEY_ID
        AWS_SECRET = self.AWS_SECRET
        ENDPOINT_URL = self.ENDPOINT_URL

        def generator_factory(file_identifier):
            s3_client = create_s3_client(AWS_ACCESS_KEY_ID, AWS_SECRET, ENDPOINT_URL)
            stream = get_file_stream(s3_client, file_identifier)
            for record in ArchiveIterator(stream, max_content_length=max_content_length):
                acc_counter.add(Counter({"n_all_records": 1}))
                try:
                    if record.headers is None:
                        acc_counter.add(Counter({"n_record_headers_none": 1}))
                        continue
                    if record.http_headers is None:
                        acc_counter.add(Counter({"n_http_headers_none": 1}))
                        continue
                    status_code = record.http_headers.status_code
                    if status_code is None:
                        acc_counter.add(Counter({"n_http_status_code_none"}))
                        # no continue because it affects a lot of websites
                        # continue
                    elif not (199 < status_code < 300):
                        acc_counter.add(Counter({"n_http_status_code_no_success"}))
                        continue
                    if record.headers['WARC-Type'] == 'response' and record.content_length >= 128:
                        content_type = str(record.http_content_type).lower()
                        if content_type.startswith("text/html"):
                            url = str(record.headers['WARC-Target-URI'])
                            html_bytes = record.reader.read()
                            try:
                                encoding = record.http_charset
                                if encoding is None:
                                    encoding = detect_encoding(html_bytes)
                                tree = HTMLTree.parse_from_bytes(html_bytes, encoding)
                            except:
                                acc_counter.add(Counter({"n_parsing_exception": 1}))
                                continue

                            # TODO: implement logic depending on if a timestamp is received or not
                            timestamp = timestamp_extractor(tree)

                            # for details see https://resiliparse.chatnoir.eu/en/stable/man/fastwarc.html#record-properties
                            if not timestamp:
                                timestamp = record.http_last_modified
                            if not timestamp:
                                timestamp = record.http_date
                            if not timestamp:
                                timestamp = record.record_date
                            timestamp = timestamp.isoformat() if timestamp else ''  # convert python date(time) object to string for serialization

                            # extracts text only from body according to
                            # https://resiliparse.chatnoir.eu/en/stable/man/extract/html2text.html?highlight=extract_plain_text
                            # -> no need to call tree.body
                            text = extract_plain_text(tree, preserve_formatting=False,
                                                      main_content=True, list_bullets=False,
                                                      alt_texts=False, links=False,
                                                      form_fields=False, noscript=False)

                            nltk.download('punkt', os.getcwd())
                            for sentence in nltk.tokenize.sent_tokenize(text):
                                acc_counter.add(Counter({"n_total_sentences": 1}))

                                if distributed_filter(sentence):
                                    acc_counter.add(Counter({"n_yielded_sentences": 1}))
                                    yield tokenizer(sentence), url, sentence, timestamp

                        else:
                            acc_counter.add(Counter({"n_wrong_content_type": 1}))
                    else:
                        acc_counter.add(Counter({"n_wrong_warc_type": 1}))
                except:
                    acc_counter.add(Counter({"n_unhandled_record_exceptions": 1}))
                    continue
            acc_counter.add(Counter({"n_finished_warc_files": 1}))

        return generator_factory

    def export(self, prediction, url, sentence, timestamp):

        with open(self.out_file, mode="a") as f:
            f.write(
                json.dumps(
                    {
                        'score': prediction.item(),
                        'url': url.decode("utf-8"),
                        'timestamp': timestamp.decode("utf-8"),
                        'sentence': sentence.decode("utf-8"),
                    },
                    ensure_ascii=False,
                )
                + '\n'
            )
