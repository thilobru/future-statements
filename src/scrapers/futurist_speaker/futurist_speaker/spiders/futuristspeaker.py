import re
from urllib.parse import unquote

import scrapy


def get_page_url(page_num):

    return f'https://futuristspeaker.com/futurist-blog/page/{page_num}/'


def get_url_page(url):

    return int(re.search(r'https://futuristspeaker.com/futurist-blog/page/(\d+)', url)[1])


class FuturistArticlesSpider(scrapy.Spider):
    
    name = "futurist_speaker"

    def start_requests(self):
        urls = [
            get_page_url(1),
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):

        current_page = get_url_page(response.request.url)

        if response.css('h2.not-found-title').get() is not None:
            return
        
        for article_card in response.css('div.et_pb_posts article'):
            article_link = article_card.css('h2.entry-title > a').attrib['href']

            yield response.follow(article_link, callback=self.parse_article)

        next_url = get_page_url(current_page + 1)
        yield response.follow(next_url, callback=self.parse)

    def parse_article(self, response):
                
        title = response.css('h1.entry-title::text').get()
        author = response.css('span.author.vcard > a::text').get()
        published = response.css('span.published::text').get()
        category = response.css('a[rel~="category"]::text').get()

        content = "".join(
            response.css('div.et_pb_text > div.et_pb_text_inner *::text').getall()
        )
        
        article = {
            'url': unquote(response.request.url),
            'title': title,
            'author': author,
            'published': published,
            'category': category,
            'content': content.strip(),
        }

        yield article
