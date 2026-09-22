"""Import tests for the source package layout."""

import importlib
import unittest


class PackageLayoutTest(unittest.TestCase):
    def test_public_packages_import(self) -> None:
        modules = [
            "ensomi_model",
            "ensomi_model.osu_core",
            "ensomi_model.features",
            "ensomi_model.data",
            "ensomi_model.events",
            "ensomi_model.timing",
            "ensomi_model.models.control",
            "ensomi_model.models.mapper.shared",
            "ensomi_model.models.mapper.v2",
            "ensomi_model.models.mapper.v2_1",
            "ensomi_model.training",
            "ensomi_model.inference",
            "ensomi_model.inference.errors",
            "ensomi_model.inference.protocol_adapter",
            "ensomi_model.inference.protobuf_transport",
            "ensomi_model.inference.service_models",
            "ensomi_model.inference.ws_framing",
            "ensomi_model.evals",
            "ensomi_model.research",
            "ensomi_model.research.scoped_style_modeling",
            "ensomi_model.research.oracle_time_continuation",
            "ensomi_model.research.bounded_typed_continuation",
        ]

        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()
