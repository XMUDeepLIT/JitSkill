import io
import base64
import json
import tempfile
import unittest
import threading
from unittest.mock import patch
from jitskill.navigator import NavController
from pathlib import Path
import torch
import numpy as np
from jitskill.explicit_record import ExplicitRecord, history_transitions, delta_text, task_instruction
from jitskill.train_navigator import build_episodes
from jitskill.latent_state import episode_forward
from jitskill.objectives import l_nav, l_alignment, l_contrast
from jitskill.fusion import SkillIndex
from jitskill.executor.projection import StateProjector
from jitskill.executor.train_projection import splice_inputs
from jitskill.executor.latent_prefix import attach
from jitskill.prompts import MEM_HEADER, SKILL_HEADER
from serving.inject_proxy import InjectionService


class CoreTests(unittest.TestCase):
    def test_record_bounds_failure_and_evidence(self):
        r = ExplicitRecord('g' * 500)
        for i in range(10):
            r.observe(f'Action: go to room {i}', 'Nothing happened' if i == 9 else 'ok' * 90)
        self.assertEqual(len(r.goal), 400)
        self.assertEqual(len(r.actions), 6)
        self.assertEqual(r.total, 10)
        self.assertLessEqual(len(r.actions[0][1]), 90)
        self.assertIn('FAILED ACTIONS:', r.render())
        r.observe('', 'object_id = 42', facts={'object_id': {'value': '42', 'evidence': 'object_id = 42'}})
        self.assertIn('object_id: 42', r.render())
        with self.assertRaises(ValueError):
            r.observe('', 'pending', subtasks={'done': {'value': 'verified', 'evidence': 'completed'}})

    def test_no_target_leakage_and_train_serve_parity(self):
        cards = [{'apis': ['app.first']}, {'apis': ['app.second']}]
        row = {'instruction': 'goal', 'scenario': 's', 'task_id': 's_1', 'steps': [
            {'say': 'step one', 'code': 'apis.app.first()', 'env_in': 'initial'},
            {'say': 'step two', 'code': 'apis.app.second()', 'env_in': 'result first'},
        ]}
        ep = build_episodes([row], cards)[0]
        self.assertEqual(ep['chunks'][0], '')
        self.assertNotIn('second', ep['chunks'][1])
        self.assertEqual(ep['marks'][1]['gold'], [1])
        msgs = [{'role': 'user', 'content': 'goal'},
                {'role': 'assistant', 'content': 'step one\n```python\napis.app.first()\n```'},
                {'role': 'user', 'content': 'result first'}]
        transitions = history_transitions(msgs)
        record = ExplicitRecord(task_instruction(msgs))
        record.observe(*transitions[0])
        self.assertEqual(ep['chunks'][1], delta_text(*transitions[0]))
        self.assertEqual(ep['marks'][1]['bt'], record.render())
        msgs[0]['content'] = 'demo'
        msgs.insert(0, {'role': 'assistant', 'content': 'example'})
        msgs.insert(2, {'role': 'user', 'content': 'now generate code to solve the actual task: goal'})
        self.assertEqual(task_instruction(msgs), 'goal')
        self.assertEqual(len(history_transitions(msgs)), 1)

    def test_recurrence_is_truncated(self):
        class Encoder:
            init_state = torch.ones(16, 2, requires_grad=True)
            previous = []
            def step(self, prev, ids):
                self.previous.append(prev.requires_grad)
                return prev * torch.tensor(2., requires_grad=True)
        e = Encoder()
        states = list(episode_forward(e, [[], [1], [2]]))
        self.assertEqual(e.previous, [False, False])
        self.assertEqual(len(states), 3)

    def test_projection_freezes_decoder_and_serializes(self):
        projector = StateProjector(3, 7)
        embedding = torch.nn.Embedding(20, 7).requires_grad_(False)
        # A causal mixing layer ensures action logits depend on preceding projected tokens.
        decoder = torch.nn.Linear(7, 20).requires_grad_(False)
        z = torch.randn(16, 3)
        x = splice_inputs(embedding, [1, 19, 2], [3, 4], 19, projector(z))
        self.assertEqual(tuple(x.shape), (1, 20, 7))
        loss = torch.nn.functional.cross_entropy(decoder(x.cumsum(1))[0, -3:-1], torch.tensor([3, 4]))
        loss.backward()
        self.assertGreater(projector.linear.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in decoder.parameters()))
        self.assertTrue(all(p.grad is None for p in embedding.parameters()))
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / 'projection.pt')
            projector.save(path)
            loaded = StateProjector.load(path)
            torch.testing.assert_close(projector(z), loaded(z))
        messages = [{'role': 'user', 'content': 'goal'}]
        part = attach(messages, projector(z))[-1]['content'][-1]
        tensor = torch.load(io.BytesIO(base64.b64decode(part['data'])), weights_only=True)
        self.assertEqual(tuple(tensor.shape), (16, 7))
        self.assertEqual(messages[-1]['content'], 'goal')

    def test_budget_and_masks(self):
        index = object.__new__(SkillIndex)
        index.pool = [{'text': 'large', 'id': 'a'}, {'text': 'small', 'id': 'b'}, {'text': 'masked', 'id': 'c'}]
        hits = index.select(np.array([3., 2., -np.inf]), k=3, budget_tokens=2,
                            count_tokens=lambda text: 3 if text == 'large' else 2)
        self.assertEqual([h['card']['id'] for h in hits], ['b'])
        self.assertEqual(index.select(np.array([-np.inf]*3), count_tokens=len), [])

    def test_losses(self):
        logits = torch.zeros(1, 3, requires_grad=True)
        self.assertAlmostEqual(l_nav(logits, [0, 2]).item(), np.log(3), places=6)
        self.assertAlmostEqual(l_alignment(logits, logits).item(), 0, places=6)
        teacher = torch.tensor([[20., -20.]], requires_grad=True)
        student = -teacher.detach().clone().requires_grad_(True)
        loss = l_alignment(teacher, student)
        self.assertGreater(loss.item(), .99)
        loss.backward()
        self.assertIsNone(teacher.grad)
        positive = torch.randn(3, requires_grad=True)
        loss = l_contrast(torch.randn(16, 3, requires_grad=True), positive, torch.randn(4, 3))
        loss.backward()
        self.assertIsNone(positive.grad)

    def test_retry_does_not_fold_again(self):
        controller = object.__new__(NavController)
        controller.state = {}
        controller.lock = threading.Lock()
        controller.init_state = torch.ones(16, 3)
        controller.dev = torch.device("cpu")
        controller.tok = lambda text, **kw: {"input_ids": [1]}
        controller.emb = torch.zeros(2, 3)
        controller.trunk = controller.slots = controller.proj = None
        controller.nav = lambda z, b: z
        with patch('jitskill.navigator.fold', side_effect=lambda *a: a[4]+1) as fold:
            _, first = controller.query('episode', ['block'], [1], return_latent=True)
            _, retry = controller.query('episode', ['block'], [1], return_latent=True)
            torch.testing.assert_close(first, retry)
            self.assertEqual(fold.call_count, 1)
            with self.assertRaises(ValueError):
                controller.query('episode', ['edited'], [1])

    def test_tool_call_evidence(self):
        messages = [{'role': 'user', 'content': 'goal'},
                    {'role': 'assistant', 'content': None,
                     'tool_calls': [{'id': '1', 'function': {'name': 'inspect', 'arguments': '{}'}}]},
                    {'role': 'tool', 'tool_call_id': '1', 'content': 'object_id = 42'}]
        blocks = history_transitions(messages)
        self.assertEqual(len(blocks), 1)
        self.assertIn('inspect', blocks[0][0])
        self.assertEqual(blocks[0][1], 'object_id = 42')

    def test_paper_prompt_and_injection_location(self):
        template = Path('prompts/injected_block.txt').read_text()
        self.assertIn(MEM_HEADER, template)
        self.assertIn(SKILL_HEADER, template)
        runtime = Path('prompts/alfworld_jitskill_executor.txt').read_text()
        self.assertIn('automatically supplies', runtime)
        self.assertNotIn('SkillRequest:', runtime)
        service = object.__new__(InjectionService)
        service.a = type('Args', (), {'injection_role': 'system'})()
        messages = [{'role': 'system', 'content': 'policy'}, {'role': 'user', 'content': 'customer'}]
        out = service.rewrite(messages, 'notes')
        self.assertEqual(out[0]['content'], 'policy\n\nnotes')
        self.assertEqual(out[1]['content'], 'customer')
        self.assertEqual(messages[0]['content'], 'policy')

if __name__ == '__main__':
    unittest.main()
