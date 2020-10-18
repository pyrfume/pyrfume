from typing import List
import json

class Publication:
    title: str = ""
    year: int = 0
    description: str = ""
    license: str = ""
    author: List[str] = []

    def __init__(self, title: str, year: int, author: List[str], description: str=""):
        self.title: str = title
        self.year: int = year
        self.description: str = description
        self.author: List[str] = author


    def json_encoder(self, obj: 'Publication'=None):
        if not obj:
            obj = self
        return json.dumps(vars(obj))

    @classmethod
    def _json_decoder(cls, json_data: dict):
        title = json_data["title"]
        year = json_data["year"]
        author = json_data["author"]
        description = json_data["description"]
        return cls(title, year, author, description)

    @classmethod
    def json_decoder(cls, json_data: str):
        return json.loads(json_data, object_hook=cls._json_decoder)
