"""Device selection tests for the inference path (run from the reconvla directory:
``python -m pytest -q tests``). The mocked tests need no accelerator."""

import os
import sys
import unittest
from contextlib import contextmanager
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recon import device as dev  # noqa: E402


@contextmanager
def fake_host(xpu=False, cuda=False):
    available = {"xpu": xpu, "cuda": cuda, "cpu": True}
    with mock.patch.object(dev, "is_available", side_effect=lambda t: available[t]):
        yield


class DeviceModuleTest(unittest.TestCase):
    def test_detection_order(self):
        with fake_host():
            self.assertEqual(dev.default_device_type(), "cpu")
            self.assertEqual(dev.resolve_device(), torch.device("cpu"))
        with fake_host(cuda=True):
            self.assertEqual(dev.resolve_device(), torch.device("cuda", 0))
        with fake_host(xpu=True, cuda=True):
            self.assertEqual(dev.resolve_device(), torch.device("xpu", 0))

    def test_explicit_device_wins(self):
        with fake_host(xpu=True):
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda", 1))
            self.assertEqual(dev.resolve_device("xpu"), torch.device("xpu", 0))
            self.assertEqual(dev.resolve_device("xpu:1"), torch.device("xpu", 1))


class LoaderTest(unittest.TestCase):
    def load(self, host, **kwargs):
        from recon.model import builder

        model = mock.MagicMock()
        model.get_vision_tower.return_value.is_loaded = False
        with fake_host(**host), mock.patch.object(builder, "AutoTokenizer"), mock.patch.object(
            builder, "ReconQwen2ForCausalLM"
        ) as cls:
            cls.from_pretrained.return_value = model
            builder.load_pretrained_model("ckpt", None, "recon", **kwargs)
        return cls.from_pretrained.call_args.kwargs, model.get_vision_tower.return_value

    def test_cpu_only_host(self):
        kwargs, tower = self.load({})
        self.assertEqual(kwargs["device_map"], {"": "cpu"})
        tower.load_model.assert_called_once_with(device_map={"": "cpu"})

    def test_xpu_host(self):
        kwargs, tower = self.load({"xpu": True})
        self.assertEqual(kwargs["device_map"], {"": "xpu:0"})
        tower.load_model.assert_called_once_with(device_map={"": "xpu:0"})

    def test_cuda_host_keeps_auto_device_map(self):
        kwargs, tower = self.load({"cuda": True})
        self.assertEqual(kwargs["device_map"], "auto")
        tower.load_model.assert_called_once_with(device_map="auto")

    def test_explicit_device_wins(self):
        kwargs, _ = self.load({"xpu": True}, device="cpu")
        self.assertEqual(kwargs["device_map"], {"": "cpu"})
        kwargs, _ = self.load({"xpu": True}, device="cuda:1")
        self.assertEqual(kwargs["device_map"], {"": "cuda:1"})

    def test_flash_attention_only_on_cuda(self):
        kwargs, _ = self.load({"xpu": True}, use_flash_attn=True)
        self.assertNotIn("attn_implementation", kwargs)
        kwargs, _ = self.load({"cuda": True}, use_flash_attn=True)
        self.assertEqual(kwargs["attn_implementation"], "flash_attention_2")


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.server = pytest_importorskip("serve.flask_server")

    def test_device_argument_defaults_to_detection(self):
        parser = self.server.build_parser()
        self.assertIsNone(parser.parse_args([]).device)
        self.assertEqual(parser.parse_args(["--device", "xpu:1"]).device, "xpu:1")

    def test_get_action_moves_inputs_to_the_model_device(self):
        robot = self.server.LLMRobotServer.__new__(self.server.LLMRobotServer)
        robot.llm_robot = mock.MagicMock()
        robot.llm_robot.device = torch.device("cpu")
        robot.llm_robot.generate.return_value = {"sequences": torch.tensor([[1, 2, 3]])}
        robot.action_tokenizer = mock.MagicMock()
        robot.action_tokenizer.decode_token_ids_to_actions.side_effect = lambda t: float(t)
        robot.temperature, robot.top_p, robot.num_beams, robot.max_new_tokens = 0.0, None, 1, 4
        actions = robot.get_action(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3, 4, 4))
        self.assertEqual(actions.tolist(), [1.0, 2.0])
        kwargs = robot.llm_robot.generate.call_args.kwargs
        self.assertEqual(kwargs["images"].dtype, torch.float16)


def pytest_importorskip(name):
    try:
        __import__(name)
    except ImportError as exc:
        raise unittest.SkipTest(f"{name} needs its inference dependencies: {exc}")
    return sys.modules[name]


@unittest.skipUnless(dev.is_available("xpu"), "needs an XPU")
class XpuParityTest(unittest.TestCase):
    def test_tiny_language_model_matches_cpu(self):
        from recon.model import ReconConfig, ReconQwen2ForCausalLM

        torch.manual_seed(0)
        config = ReconConfig(
            hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=128, max_position_embeddings=64,
        )
        model = ReconQwen2ForCausalLM(config).to(torch.bfloat16).eval()
        with torch.no_grad():
            for p in model.parameters():  # no layer may stay zero-initialised
                p.copy_(torch.randn_like(p.float()).to(p.dtype) * 0.05)
        embeds = torch.randn(1, 12, 64).to(torch.bfloat16)
        with torch.no_grad():
            ref = model(inputs_embeds=embeds).logits.float()
            out = model.to("xpu")(inputs_embeds=embeds.to("xpu")).logits.float().cpu()
        self.assertGreater(ref.std().item(), 0)
        torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    unittest.main()
