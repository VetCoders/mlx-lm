"""W1 contracts: authored without imports, collection or execution during embargo."""

import copy
import os
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models import cache as shared


class RegistryTestSpecializedCache(shared._BaseCache):
    """Minimal external codec; constructor is deliberately not a restore path."""

    def __init__(self, values, label):
        self.values = values
        self.label = label

    @property
    def state(self):
        return [self.values]

    @state.setter
    def state(self, value):
        self.values = value[0]

    @property
    def meta_state(self):
        return self.label

    @meta_state.setter
    def meta_state(self, value):
        self.label = value


def tokens(values):
    return mx.array(values, dtype=mx.float32).reshape(1, 1, -1, 1)


class TestCacheTypeRegistry(unittest.TestCase):
    def setUp(self):
        # Restore only this private codec table, never shared classes/methods.
        self.registry = patch.dict(shared._REGISTERED_CACHE_TYPES)
        self.registry.start()
        self.addCleanup(self.registry.stop)

    def register_specialized(self):
        shared.register_cache_type(RegistryTestSpecializedCache)
        return RegistryTestSpecializedCache(tokens([7, 8]), "pool-fixture")

    def assert_history(self, array, expected):
        self.assertEqual(array.reshape(-1).tolist(), expected)

    def test_canonical_namespace_and_roundtrip(self):
        names = (
            "_BaseCache",
            "KVCache",
            "ArraysCache",
            "BatchKVCache",
            "RotatingKVCache",
            "BatchRotatingKVCache",
            "CacheList",
            "ChunkedKVCache",
            "QuantizedKVCache",
            "ConcatenateKVCache",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertIs(shared._resolve_cache_type(name), getattr(shared, name))
        caches = [
            shared.KVCache(),
            shared.RotatingKVCache(8),
            shared.ChunkedKVCache(8),
            shared.ConcatenateKVCache(),
        ]
        for cache in caches:
            cache.update_and_fetch(tokens([1, 2, 3]), tokens([11, 12, 13]))
        arrays = shared.ArraysCache(1)
        arrays[0] = tokens([21, 22])
        source = shared.CacheList(*caches, arrays)
        restored = shared.CacheList.from_state(
            copy.deepcopy(source.state), copy.deepcopy(source.meta_state)
        )
        for old, new in zip(source.caches, restored.caches):
            self.assertIs(type(new), type(old))
            self.assertIsNot(new, old)
            self.assertEqual(new.meta_state, old.meta_state)
            self.assertTrue(mx.array_equal(new.state[0], old.state[0]))
        restored[0].keys[..., 0, :] = 99
        self.assert_history(source[0].state[0], [1, 2, 3])

    def test_nested_registered_raw_roundtrip_and_caller_detachment(self):
        special = self.register_specialized()
        ordinary = shared.ConcatenateKVCache()
        ordinary.update_and_fetch(tokens([1, 2]), tokens([11, 12]))
        source = shared.CacheList(ordinary, shared.CacheList(special))
        self.assertEqual(source.meta_state[0], ["ConcatenateKVCache", "CacheList"])
        restored = shared.CacheList.from_state(
            copy.deepcopy(source.state), copy.deepcopy(source.meta_state)
        )
        self.assertIs(type(restored), shared.CacheList)
        self.assertIs(type(restored[0]), shared.ConcatenateKVCache)
        self.assertIs(type(restored[1]), shared.CacheList)
        self.assertIs(type(restored[1][0]), RegistryTestSpecializedCache)
        self.assertEqual(restored[1][0].label, "pool-fixture")
        self.assert_history(restored[1][0].values, [7, 8])
        restored[1][0].values[..., 0, :] = 70
        self.assert_history(special.values, [7, 8])
        restored[0].trim(1)
        restored[0].update_and_fetch(tokens([9]), tokens([19]))
        self.assert_history(restored[0].keys, [1, 9])
        self.assert_history(ordinary.keys, [1, 2])
        self.assertNotIn("RegistryTestSpecializedCache", vars(shared))

    def test_real_file_uses_registered_top_level_and_nested_decoders(self):
        special = self.register_specialized()
        ordinary = shared.ConcatenateKVCache()
        ordinary.update_and_fetch(tokens([2, 3]), tokens([12, 13]))
        source = [special, shared.CacheList(ordinary, shared.CacheList(special))]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tiny.safetensors")
            shared.save_prompt_cache(path, source, {"fixture": "no-weights"})
            restored, metadata = shared.load_prompt_cache(path, return_metadata=True)
            plain = shared.load_prompt_cache(path)
        self.assertEqual(metadata, {"fixture": "no-weights"})
        self.assertIs(type(restored[0]), RegistryTestSpecializedCache)
        self.assertIs(type(plain[0]), RegistryTestSpecializedCache)
        self.assertIs(type(restored[1][0]), shared.ConcatenateKVCache)
        self.assertIs(type(restored[1][1][0]), RegistryTestSpecializedCache)
        self.assert_history(restored[1][0].values, [12, 13])
        self.assert_history(restored[1][1][0].values, [7, 8])
        restored[0].values[..., 0, :] = 700
        self.assert_history(special.values, [7, 8])
        self.assert_history(restored[1][1][0].values, [7, 8])

    def test_registration_idempotency_and_failure_atomicity(self):
        self.register_specialized()
        before = dict(shared._REGISTERED_CACHE_TYPES)
        shared.register_cache_type(RegistryTestSpecializedCache)
        self.assertEqual(shared._REGISTERED_CACHE_TYPES, before)
        invalid = [
            None,
            object,
            shared._BaseCache,
            RegistryTestSpecializedCache([], "x"),
            type("InvalidCodec", (shared._BaseCache,), {"from_state": None}),
        ]
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                with self.assertRaises(TypeError):
                    shared.register_cache_type(candidate)
                self.assertEqual(shared._REGISTERED_CACHE_TYPES, before)
        for name in (
            "KVCache",
            "ConcatenateKVCache",
            "mx",
            "copy",
            "TokenBuffer",
            "register_cache_type",
            "list",
            "eval",
            "__builtins__",
            "RegistryTestSpecializedCache",
        ):
            candidate = type(name, (shared._BaseCache,), {})
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    shared.register_cache_type(candidate)
                self.assertEqual(shared._REGISTERED_CACHE_TYPES, before)
        with self.assertRaises(ValueError):
            shared.register_cache_type(shared.ConcatenateKVCache)
        self.assertEqual(shared._REGISTERED_CACHE_TYPES, before)
        self.assertIs(
            shared._resolve_cache_type("ConcatenateKVCache"), shared.ConcatenateKVCache
        )

    def test_unknown_names_do_not_decode_or_mutate_registry(self):
        self.register_specialized()
        before = dict(shared._REGISTERED_CACHE_TYPES)
        for name in (
            "RegistryMissingCache",
            "mx",
            "TokenBuffer",
            "eval",
            "some.module.Cache",
            "__import__('os')",
            None,
            [],
            42,
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    shared.CacheList.from_state([[]], ([name], [""]))
                self.assertEqual(shared._REGISTERED_CACHE_TYPES, before)

    def test_parallel_arities_refused_before_any_child_decode(self):
        calls = []

        class RegistryCountingCache(shared._BaseCache):
            @classmethod
            def from_state(cls, state, meta_state):
                calls.append(state)
                return super().from_state(state, meta_state)

        shared.register_cache_type(RegistryCountingCache)
        name = RegistryCountingCache.__name__
        before = dict(shared._REGISTERED_CACHE_TYPES)
        cases = [
            ([[], []], ([name], [""])),
            ([[]], ([name, name], [""])),
            ([[]], ([name], ["", ""])),
            ([[]], ([name], [])),
            ([[]], ([name],)),
            ([[]], ([name], [""], [])),
            ([[]], (name, [""])),
            (None, ([name], [""])),
            ([[], []], ([name, "UnknownCodec"], ["", ""])),
        ]
        for state, metadata in cases:
            with self.subTest(state=state, metadata=metadata):
                with self.assertRaises(ValueError):
                    shared.CacheList.from_state(state, metadata)
                self.assertEqual(calls, [])
                self.assertEqual(shared._REGISTERED_CACHE_TYPES, before)
        self.assertEqual(shared.CacheList.from_state([], ([], [])).caches, [])

    def test_malformed_file_arities_and_unknown_names(self):
        self.register_specialized()
        before = dict(shared._REGISTERED_CACHE_TYPES)
        # Write actual tiny files with the existing flat safetensors wire format.
        state = [[tokens([1])]]
        cases = [
            [["x", "y"], {}, ["RegistryTestSpecializedCache"]],
            [["x"], {}, ["RegistryTestSpecializedCache", "KVCache"]],
            [["x"], {}, ["UnknownFileCodec"]],
            [["x"], {}, ["RegistryTestSpecializedCache"], "extra"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, metadata in enumerate(cases):
                path = os.path.join(directory, f"invalid-{index}.safetensors")
                mx.save_safetensors(
                    path, dict(tree_flatten(state)), dict(tree_flatten(metadata))
                )
                with self.subTest(index=index):
                    with self.assertRaises(ValueError):
                        shared.load_prompt_cache(path)
                    self.assertEqual(shared._REGISTERED_CACHE_TYPES, before)

    def test_concatenate_empty_restore_then_append(self):
        source = shared.ConcatenateKVCache()
        restored = shared.ConcatenateKVCache.from_state(source.state, source.meta_state)
        self.assertIs(type(restored), shared.ConcatenateKVCache)
        self.assertIsNot(source, restored)
        self.assertEqual(restored.state, (None, None))
        self.assertEqual(restored.offset, 0)
        self.assertEqual(restored.trim(4), 0)
        self.assertTrue(restored.empty())
        restored.update_and_fetch(tokens([5, 6]), tokens([15, 16]))
        self.assertEqual(restored.offset, 2)
        self.assert_history(restored.keys, [5, 6])
        self.assert_history(restored.values, [15, 16])
        self.assertEqual(source.state, (None, None))

    def test_concatenate_trim_append_never_revives_history(self):
        cache = shared.ConcatenateKVCache()
        cache.update_and_fetch(tokens([1, 2, 3, 4]), tokens([11, 12, 13, 14]))
        self.assertEqual(cache.trim(2), 2)
        self.assertEqual(cache.offset, 2)
        self.assert_history(cache.keys, [1, 2])
        self.assert_history(cache.values, [11, 12])
        cache.update_and_fetch(tokens([8, 9]), tokens([18, 19]))
        self.assertEqual(cache.offset, 4)
        self.assert_history(cache.keys, [1, 2, 8, 9])
        self.assert_history(cache.values, [11, 12, 18, 19])
        self.assertEqual(cache.trim(99), 4)
        self.assertEqual(cache.keys.shape[-2], 0)
        self.assertEqual(cache.values.shape[-2], 0)
        cache.update_and_fetch(tokens([7]), tokens([17]))
        self.assertEqual(cache.offset, 1)
        self.assert_history(cache.keys, [7])
        self.assert_history(cache.values, [17])


if __name__ == "__main__":
    unittest.main()
