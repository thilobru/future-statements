import re
import os
import json
from collections import Counter
import random
from pathlib import Path
from datetime import datetime

import nltk
import numpy as np
import tensorflow as tf
nltk.download('punkt')
from fastwarc.warc import ArchiveIterator
from resiliparse.parse.lang import detect_fast
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse import detect_encoding
from resiliparse.parse.html import HTMLTree

from pipelines.text_pipeline import TextPipeline
from pipelines.tools.passthrough_model import PassthroughModelPipeline
from helpers import create_s3_client, get_file_stream


# parent classes searched from left to right -> should work
class FutureRegexPipeline(TextPipeline, PassthroughModelPipeline):
    """
    
    """
    
    def __init__(self, output_dir, regex, min_length, max_length):
        self.regex = regex
        self.min_length = min_length
        self.max_length = max_length
        max_content_length=4000000
        out_dir = Path(output_dir)

        super().__init__(out_dir=out_dir, max_content_length=max_content_length)

    def get_distributed_filter(self):
        regex = self.regex

        def distributed_filter(text):
            return re.search(regex, text.lower()) is not None

        return distributed_filter

    def get_base_filter(self):
        min_length = self.min_length
        max_length = self.max_length

        def base_filter(text):

            if len(text) < min_length:
                return False
            
            if len(text) > max_length:
                return False

            if not detect_fast(text)[0] == "en":
                return False

            return True

        return base_filter

    def get_signature(self):
        return (
            self.get_tokens_spec(),  # text for classification
            tf.TensorSpec(shape=(), dtype=tf.string),  # text for export
            tf.TensorSpec(shape=(), dtype=tf.string),  # url
            tf.TensorSpec(shape=(), dtype=tf.bool), # label
        )

    def get_generator_factory(self):
        acc_counter = self.acc_counter
        max_content_length = self.max_content_length
        distributed_filter = self.get_distributed_filter()
        base_filter = self.get_base_filter()
        tokenizer = self.get_tokenizer()
        AWS_ACCESS_KEY_ID = self.AWS_ACCESS_KEY_ID
        AWS_SECRET = self.AWS_SECRET
        ENDPOINT_URL = self.ENDPOINT_URL

        def generator_factory(file_identifier):
            s3_client = create_s3_client(AWS_ACCESS_KEY_ID, AWS_SECRET, ENDPOINT_URL)
            stream = get_file_stream(s3_client, file_identifier)
            for record in ArchiveIterator(stream, max_content_length=max_content_length):
                acc_counter.add(Counter({"n_all": 1}))
                try:
                    if record.headers is None:
                        acc_counter.add(Counter({"n_record_headers_none": 1}))
                        continue
                    if record.http_headers is None:
                        acc_counter.add(Counter({"n_http_headers_none": 1}))
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
                            
                            text = extract_plain_text(tree, preserve_formatting=False,
                                                        main_content=True, list_bullets=False,
                                                        alt_texts=False, links=False,
                                                        form_fields=False, noscript=False)

                            nltk.download('punkt', os.getcwd())
                            
                            sentences = {
                                'future': [],
                                'no_future': [],
                            }
                            for sentence in nltk.tokenize.sent_tokenize(text):
                                acc_counter.add(Counter({"sentence": 1}))

                                if base_filter(sentence):
                                    
                                    if distributed_filter(sentence):
                                        sentences['future'].append(sentence)
                                    else:
                                        sentences['no_future'].append(sentence)

                            if sentences['future']:
                                sentence = random.choice(sentences['future'])
                                acc_counter.add(Counter({"n_future": 1}))
                                yield sentence, sentence, url, True

                            elif sentences['no_future']:
                                sentence = random.choice(sentences['no_future'])
                                acc_counter.add(Counter({"n_no_future": 1}))
                                yield sentence, sentence, url, False
                            else:
                                acc_counter.add(Counter({"n_no_sentences": 1}))
                        else:
                            acc_counter.add(Counter({"n_wrong_content_type": 1}))
                    else:
                        acc_counter.add(Counter({"n_wrong_warc_type": 1}))
                except Exception:
                    acc_counter.add(Counter({"n_unhandled_record_exceptions": 1}))
                    continue
            acc_counter.add(Counter({"n_finished_warc_files": 1}))

        return generator_factory

    def export(self, _, sentence, url, is_future):
        
        url = url.decode("utf-8")
        is_future = bool(is_future)
        out_file = 'positive.jsonl' if is_future else 'negative.jsonl'
        out_file = self.out_dir / out_file
        with open(out_file, mode='a') as f:
            f.write(
                json.dumps(
                    {
                        'url': url,
                        'text': sentence.decode("utf-8"),
                        'is_future': is_future,
                    },
                ) + '\n'
            )


if __name__ == "__main__":

    out_dir = Path(__file__).parent.parent / "data-{}".format(datetime.now().isoformat())
    min_length = 30
    max_length = 300

    verbs = r"|".join(
        [
            r"(?<!\bi )(?<!\byou )(will|won(’|')t|will not)\b",
            r"\bthere(’|')ll\b",
            r"\bthey(’|')ll\b",
            r"\bwe(’|')ll",
            r"\bgoing to\b",
            r"\bpredict(ing|s|\b)",
            r"\bexpect(ing|s|\b)",
            r"\bforecast(ing|s|\b)",
            r"\bestimat(ing|es|e\b)",
            r"\bpresum(ing|es|e\b)",
            r"\bspeculat(ing|es|e\b)",
            r"\benvision(ing|s|\b)",
            r"\bimagin(ing|es|e\b)",
            r"\bforetell(ing|s|\b)",
            r"\bbeliev(ing|es|e\b)",
            r"\banticipat(ing|es|e\b)"
        ]
    )
    fill_words = r"([a-zA-Z0-9\.,%\$€\&\-\"\'\(\)\s]*)"
    expressions = r"|".join(
            [
                r"\b(in|for) the future\b",
                r"\b(in|for) the foreseeable future\b",
                r"\b(in|for) the not too distant future\b",
                r"\b(in|for) the near future\b",
                r"\b(in|for) the distant future\b",
                r"\b(in|within) a few years\b",
                r"\b(in|for|within) the next [0-9\s]{0,3}years\b",
                r"\b(in|for|within) the next [a-zA-Z]+ years\b",
                r"\b(in|for|within) the next few years\b",
                r"\b(in|for|within) the years ahead\b",
                r"\b(in|within) [0-9]{1,3} years\b",
                r"\bby [0-9]{4}\b",
                r"\bsomeday\b",
                r"\bin the long run\b",
            ]
        )
    regex = "^" + fill_words + \
        "(((" + verbs + ")" + fill_words + "(" + expressions + "))" + "|" + \
        "((" + expressions + ")" + fill_words + "(" + verbs + ")))" + \
        fill_words + "$"

    p = FutureRegexPipeline(
        output_dir=out_dir,
        min_length=min_length,
        max_length=max_length,
        regex=regex,
    )
    p.run()
