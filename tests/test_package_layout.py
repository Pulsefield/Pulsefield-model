"""Import tests for the source package layout."""

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
            "pulsefield_model.models.mapper.shared",
            "pulsefield_model.models.mapper.v2",
            "pulsefield_model.models.mapper.v2_1",
            "pulsefield_model.training",
            "pulsefield_model.inference",
            "pulsefield_model.inference.errors",
            "pulsefield_model.inference.protocol_adapter",
            "pulsefield_model.inference.protobuf_transport",
            "pulsefield_model.inference.service_models",
            "pulsefield_model.inference.ws_framing",
            "pulsefield_model.evals",
            "pulsefield_model.research",
            "pulsefield_model.research.scoped_style_modeling",
            "pulsefield_model.research.oracle_time_continuation",
            "pulsefield_model.research.bounded_typed_continuation",
        ]

        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()
