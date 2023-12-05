# Copyright (c) 2023, DeepLink.

import os
from typing import Dict


class local_eviron:
    def __init__(self, new_envrons: Dict[str, str]):
        self.new_envrons = new_envrons
        self.old_envrons = {}

    def __enter__(self):
        for var, new_value in self.new_envrons.items():
            self.old_envrons[var] = os.environ.get(var, None)
            os.environ[var] = new_value

    def __exit__(self, exc_type, exc_value, traceback):
        for var, old_value in self.old_envrons.items():
            if old_value is None:
                del os.environ[var]
            else:
                os.environ[var] = old_value
