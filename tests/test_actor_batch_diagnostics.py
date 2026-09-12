import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from shopping_grpo.training.grpo.dynamic_sampling import append_training_diagnostic, cuda_memory_snapshot, record_actor_batch_pre_update

class TestActorBatchDiagnostics(unittest.TestCase):
    def make_batch(self, response_mask=True, attention=True):
        tensors = {}
        if attention: tensors["attention_mask"] = torch.tensor([[1,1,1,1],[1,1,0,0]])
        tensors["responses"] = torch.tensor([[31,32],[41,0]])
        if response_mask: tensors["response_mask"] = torch.tensor([[1,1],[1,0]])
        return SimpleNamespace(batch=tensors, non_tensor_batch={"uid":["u1","u2"],"task_id":[7,8]})

    def test_response_mask_is_derived_from_attention_suffix(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"d.jsonl"; record_actor_batch_pre_update(self.make_batch(False),p,2); x=json.loads(p.read_text())
        self.assertEqual(x["availability"], "available")
        self.assertEqual(x["response_mask_source"], "attention_mask_suffix")
        self.assertEqual(x["rows"][0], {"attention_tokens":4,"prompt_tokens":2,"response_tokens":2})
        self.assertEqual(x["batch_shapes"]["responses"], [2,2])

    def test_missing_attention_degrades_safely(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"d.jsonl"; record_actor_batch_pre_update(self.make_batch(attention=False),p,2); x=json.loads(p.read_text())
        self.assertEqual(x["availability"], "unavailable")
        self.assertEqual(x["reason"], "missing_attention_mask")
        self.assertEqual(x["rows"], [])

    def test_cpu_snapshot_and_append_schema(self):
        self.assertFalse(cuda_memory_snapshot("cpu")["cuda_available"])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"d.jsonl"; append_training_diagnostic(p,"x",3,foo=1); x=json.loads(p.read_text())
        self.assertEqual(x["schema_version"],1); self.assertEqual(x["event"],"x"); self.assertEqual(x["foo"],1)
