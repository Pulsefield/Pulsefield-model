"""Smoke tests for the source package layout."""

import importlib
import unittest


class PackageLayoutTest(unittest.TestCase):
    def test_public_packages_import(self) -> None:
        modules = [
            "pulsefield_model",
            "pulsefield_model.osu_core",
            "pulsefield_model.features",
            "pulsefield_model.data",
            "pulsefield_model.events",
            "pulsefield_model.timing",
            "pulsefield_model.models.control",
            "pulsefield_model.models.mapper.v1",
            "pulsefield_model.models.mapper.v2",
            "pulsefield_model.models.mapper.v2_1",
            "pulsefield_model.training",
            "pulsefield_model.inference",
            "pulsefield_model.evals",
        ]

        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()
