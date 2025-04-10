import json
import pathlib
import requests
from requests.structures import CaseInsensitiveDict

url = "https://www.elon.edu/u/imagining/wp-admin/admin-ajax.php"

headers = CaseInsensitiveDict()
headers["User-Agent"] = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:101.0) Gecko/20100101 Firefox/101.0"
headers["Accept"] = "application/json, text/javascript, */*; q=0.01"
headers["Accept-Language"] = "de,en-US;q=0.7,en;q=0.3"
headers["Accept-Encoding"] = "gzip, deflate, br"
headers["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"
headers["X-Requested-With"] = "XMLHttpRequest"
headers["Origin"] = "https://www.elon.edu"
headers["Connection"] = "keep-alive"
headers["Referer"] = "https://www.elon.edu/u/imagining/time-capsule/early-90s/90s-database/"
headers["Sec-Fetch-Dest"] = "empty"
headers["Sec-Fetch-Mode"] = "cors"
headers["Sec-Fetch-Site"] = "same-origin"

def get_data_str(page_num):
    return f"keywords=&predictor=&expertise=0&topic=0&subtopic=0&prediction_date=&publication_name=&title=&publication_author=&prediction_medium=0&publication_date=&excerpt=&content=&action=experts_search&page={page_num}&ppp=30"


with open(pathlib.Path(__file__).parent / 'internet_future_data.jsonl', mode='w') as out_f:

    page_num = 1
    resp = requests.post(url, headers=headers, data=get_data_str(page_num=page_num))

    while(resp.status_code == 200 and (resp_data := resp.json())):

        print(page_num)

        for datapoint in resp_data:
            if 'posts_found' in datapoint:
                continue
            else:
                out_f.write(json.dumps(datapoint) + '\n')

        page_num += 1
        resp = requests.post(url, headers=headers, data=get_data_str(page_num=page_num))
