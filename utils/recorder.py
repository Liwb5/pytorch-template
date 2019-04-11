# coding:utf-8
import json
import logging

class Recorder:
    """
    Training process recorder 

    Note:
        Used by BaseTrainer to record training history.
    """
    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)


