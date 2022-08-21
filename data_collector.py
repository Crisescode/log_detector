# coding: utf-8

from elasticsearch import Elasticsearch
from elasticsearch import helpers


class DataCollectorWithKafka:
    pass


class DataCollectorWithElasticSearch:
    def __init__(self, hosts: str, index: str, doc_type: str):
        self.hosts = hosts
        self.index = index
        self.doc_type = doc_type
        self.es_client = Elasticsearch(hosts=self.hosts)

    def _search(self, payload: dict):
        return self.es_client.search(
            index=self.index,
            doc_type=self.doc_type,
            body=payload
        )["hits"]["hits"]

    def _scan(
            self,
            payload: dict,
            size: int = 10000,
            scroll: str = "5m",
            timeout: int = "1m"
    ):
        return helpers.scan(
            client=self.es_client,
            index=self.index,
            doc_type=self.doc_type,
            query=payload,
            scroll=scroll,
            size=size,
            timeout=timeout
        )

    def get_log(self):
        # todo
        pass
