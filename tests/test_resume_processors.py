import pytest

from drtc.compat import resume_processors


def test_unmatched_normalizer_overrides_are_ignored_others_still_raise():
    from lerobot.processor.pipeline import PolicyProcessorPipeline

    resume_processors.install()
    resume_processors.install()  # idempotent
    cfg = {"steps": [{"registry_name": "groot_n15_pack_inputs_v3"}, {"registry_name": "device_processor"}]}
    PolicyProcessorPipeline._validate_overrides_used({"normalizer_processor"}, cfg)
    PolicyProcessorPipeline._validate_overrides_used({"unnormalizer_processor"}, cfg)
    with pytest.raises(KeyError, match="typo_processor"):
        PolicyProcessorPipeline._validate_overrides_used({"normalizer_processor", "typo_processor"}, cfg)
