from typing import Optional

from .logger import Logger

class SrtLabelerPipeline:
    def __init__(
        self,
        api_key: Optional[str] = None,
        domain: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        self.log = Logger(self.__class__.__name__, debug=debug)
        self.api_key = api_key
        self.domain = domain
        self.debug = debug
