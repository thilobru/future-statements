from email.policy import strict
import re
import nltk
import json
from pathlib import Path
from resiliparse.parse.lang import detect_fast
from argparse import ArgumentParser


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--scraped_data_file", type=Path, required=True)
    parser.add_argument("--out_data_dir", type=Path, required=True)
    parser.add_argument("--data_source", choices=['futurist', 'nineties'], type=str, required=True)
    parser.add_argument("--regex", choices=['strict', 'loose'], type=str)

    return parser.parse_args()


def _distributed_filter(text, regex):
    return re.search(regex, text.lower()) is not None


def _base_filter(text, min_length, max_length):

    if len(text) < min_length:
        return False

    if len(text) > max_length:
        return False

    if not detect_fast(text)[0] == "en":
        return False

    return True


def _read_data_from_json(file_path, data_source):
    with open(file_path, mode='r', encoding='utf-8') as f:
        for i, json_obj in enumerate(f):
            obj = json.loads(json_obj) #, strict=False)
            if data_source == 'futurist':
                if 'content' in obj.keys(): 
                    yield obj['url'], obj['content']
                else: print(obj)
            elif data_source == 'nineties':
                if 'permalink' in obj.keys():
                    yield obj['permalink'], obj['excerpt']
                else: print(obj)


def _strict_regex():
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
    
    return regex

def _loose_regex():
    verbs = r"|".join(
        [
            r"(?<!\bi )(?<!\byou )(will|won(’|')t|will not)\b",
            r"\bthere(’|')ll\b",
            r"\bthey(’|')ll\b",
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
    fill_words = r"([a-zA-Z0-9\.,%\$€\&\-\"\'\(\)\s]*)"
    regex = "((" + expressions + ")" + "|" + "(" + verbs + "))" + fill_words + "$"
    return regex

if __name__ == "__main__":
    args = vars(_parse_args())
    print('\n\nargs: ', args)
    nltk.download('punkt')

    min_length = 30
    max_length = 300
    regex = _strict_regex()
    if args['regex'] == 'loose':
        regex = _loose_regex()

    in_path = args['scraped_data_file']

    future_statements = []
    no_future_statements = []
    for url, text in _read_data_from_json(in_path, args['data_source']):
        for s in nltk.tokenize.sent_tokenize(text):
            for sentence in s.split('\n'):
                if _base_filter(sentence, min_length, max_length):
                    if _distributed_filter(sentence, regex):
                        future_statements.append({'url': url, 'text': sentence, 'is_future': 1})
                    else:
                        no_future_statements.append({'url': url, 'text': sentence, 'is_future': 0})

    out_dir = args['out_data_dir']
    positive_path = out_dir / 'positive.jsonl'
    with open(positive_path, 'w') as f:
        for item in future_statements:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    negative_path = out_dir / 'negative.jsonl'
    with open(negative_path, 'w') as f:
        for item in no_future_statements:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
