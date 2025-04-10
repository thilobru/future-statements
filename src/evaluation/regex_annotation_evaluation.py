import json
from pathlib import Path
from regex_classification import get_metrics

if __name__ == "__main__":
    min_length = 30
    max_length = 300

    file_path = Path(__file__).parent.parent / 'annotated_regex_data_00010.jsonl'

    classified_statements = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        for i, json_obj in enumerate(f):
            obj = json.loads(json_obj)
            print(obj)
            s = obj['text']
            if len(obj['label']) == 1:
                if obj['label'][0] == 'future':
                    classified_statements.append({'text': s, 'true_label': 1, 'pred_label': 1})
                elif obj['label'][0] == 'no_future':
                    classified_statements.append({'text': s, 'true_label': 0, 'pred_label': 1})

    
    metrics = get_metrics(classified_statements)
    
    print(metrics)