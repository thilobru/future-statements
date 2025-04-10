import re
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score 
from resiliparse.parse.lang import detect_fast


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


def get_metrics(classified_statements):
    y_true = [d['true_label'] for d in classified_statements]
    y_pred = [d['pred_label'] for d in classified_statements]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    tp = len([1 for d in classified_statements if d['pred_label'] == 1 and d['true_label'] == 1])
    fp = len([1 for d in classified_statements if d['pred_label'] == 1 and d['true_label'] == 0])
    tn = len([1 for d in classified_statements if d['pred_label'] == 0 and d['true_label'] == 0])
    fn = len([1 for d in classified_statements if d['pred_label'] == 0 and d['true_label'] == 1])

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


if __name__ == "__main__":
    min_length = 30
    max_length = 300
    loose_regex = _loose_regex()
    strict_regex = _strict_regex()

    file_path = Path(__file__).parent.parent / 'ml/datasets/evaluation_data.jsonl'

    loose_statements = []
    strict_statements = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        for i, json_obj in enumerate(f):
            obj = json.loads(json_obj)
            s = obj['text']
            if _base_filter(s, min_length, max_length):
                if _distributed_filter(s, loose_regex):
                    loose_statements.append({'text': s, 'true_label': obj['is_future'], 'pred_label': 1})
                else:
                    loose_statements.append({'text': s, 'true_label': obj['is_future'], 'pred_label': 0})
                if _distributed_filter(s, strict_regex):
                    strict_statements.append({'text': s, 'true_label': obj['is_future'], 'pred_label': 1})
                else:
                    strict_statements.append({'text': s, 'true_label': obj['is_future'], 'pred_label': 0})

    
    loose_metrics = get_metrics(loose_statements)
    print('loose regex:\n', loose_metrics)
    strict_metrics = get_metrics(strict_statements)
    print('\nstrict regex:\n', strict_metrics)