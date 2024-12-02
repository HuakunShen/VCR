import os
import scrapy
import pandas as pd

from scrapy.crawler import CrawlerProcess


class CrawlModels(scrapy.Spider):
    name = "CrawlModels"

    def start_requests(self):
        urls = [
            'https://pytorch.org/vision/stable/models.html'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        lis = response.css(
            "#classification > div.toctree-wrapper.compound > ul > li")
        alinks = [li.css("a") for li in lis]
        hrefs = [
            f'https://pytorch.org/vision/stable/{alink.attrib["href"]}' for alink in alinks]
        for href in hrefs:
            table = pd.read_html(href)
            model_names = table[0][0].to_list()
            model_names = [m.split("(")[0] for m in model_names]
            for model_name in model_names:
                yield {"model_name": model_name}


if __name__ == "__main__":
    process = CrawlerProcess(settings={
        "FEEDS": {
            "models.csv": {"format": "csv"},
        },
    })
    process.crawl(CrawlModels)
    process.start()
    model_df = pd.read_csv("models.csv")
    os.remove("models.csv")
    for model_name in model_df["model_name"].to_list():
        print(f'"{model_name}": models.{model_name},')
