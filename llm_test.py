import unittest
import os, sys
from pathlib import Path
from omegaconf import OmegaConf as Conf

__dir_path__ = Path(os.path.realpath(__file__)).parent


class ChatTest(unittest.TestCase):

    def setUp(self):
        pass
        self.cfg = Conf.create()
        self.cfg = Conf.merge(
            self.cfg, Conf.load(__dir_path__.joinpath("cfg.settings.json"))
        )
        self.cfg = Conf.merge(
            self.cfg, Conf.load(__dir_path__.joinpath("cfg.secrets.json"))
        )

    def tearDown(self):
        pass

    def test_text_classifier(self):
        # Assert pre-conditions
        self.assertIsNotNone(self.cfg)

    def test_text_solver(self):
        # Assert pre-conditions
        self.assertIsNotNone(self.cfg)


if __name__ == "__main__":
    unittest.main()
