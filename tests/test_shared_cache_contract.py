"""Shared LM cache contracts; tiny deterministic arrays, no model artifacts.

Authored under W1 compile embargo. Execute only after integrator W2 closure.
Behavioral provenance: VLM 1068e538 cache deltas and oMLX 94530d8 None guard.
"""

import copy
import unittest

import mlx.core as mx

from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)


def history(rows):
    return mx.array(rows, dtype=mx.float32)[:, None, :, None]


class TestSharedCacheContract(unittest.TestCase):
    def assert_array(self, actual, expected):
        mx.eval(actual)
        self.assertEqual(actual.tolist(), expected)

    def assert_history(self, cache, expected):
        self.assert_array(cache.keys[..., : cache.offset, :].reshape(-1), expected)
        self.assert_array(cache.values[..., : cache.offset, :].reshape(-1),
                          [v + 100 for v in expected])

    def test_kv_extract_copies_each_row_and_logical_history(self):
        source = KVCache()
        x = history([[11, 12, 13], [21, 22, 23]])
        source.update_and_fetch(x, x + 100)
        source.trim(1)
        for index, expected in [(0, [11, 12]), (1, [21, 22]),
                                (-1, [21, 22]), (-2, [11, 12])]:
            with self.subTest(index=index):
                row = source.extract(index)
                self.assertIs(type(row), KVCache)
                self.assertIsNot(row, source)
                self.assertIsNot(row.keys, source.keys)
                self.assertEqual(row.keys.shape, (1, 1, 2, 1))
                self.assert_history(row, expected)
                row.keys[0, 0, 0, 0] = -7
                row.values[0, 0, 0, 0] = -9
                self.assert_array(source.keys[:, 0, :2, 0], [[11, 12], [21, 22]])
                self.assert_array(source.values[:, 0, :2, 0], [[111, 112], [121, 122]])
        before = source.keys
        for index in (-3, 2):
            with self.assertRaises(IndexError):
                source.extract(index)
            self.assertIs(source.keys, before)
            self.assertEqual(source.offset, 2)

    def test_singleton_empty_and_trimmed_zero_extraction(self):
        empty = KVCache()
        for index in (0, -1):
            row = empty.extract(index)
            self.assertIsNot(row, empty)
            self.assertTrue(row.empty())
            self.assertEqual(row.offset, 0)
        for index in (-2, 1):
            with self.assertRaises(IndexError):
                empty.extract(index)
        x = history([[7]])
        empty.update_and_fetch(x, x + 100)
        row = empty.extract(0)
        row.keys[0, 0, 0, 0] = 99
        self.assert_history(empty, [7])
        empty.trim(1)
        row = empty.extract(-1)
        self.assertEqual(row.keys.shape, (1, 1, 0, 1))
        self.assertEqual(row.offset, 0)

    def test_arrays_pending_filter_extract_join_and_isolation(self):
        for filled in (False, True):
            with self.subTest(filled=filled):
                source = ArraysCache(3, left_padding=[2, 0, 1])
                source.prepare(lengths=[5, 8, 6])
                if filled:
                    # Slot zero absent is legal; never infer all-empty from it.
                    source[1] = mx.array([[10, 11], [20, 21], [30, 31]])
                source.filter([2, 0])
                last = source.extract(-1)
                first = source.extract(0)
                self.assertEqual(last.batch_size, 1)
                self.assertIsNone(last[0])
                self.assertIsNone(last[2])
                self.assert_array(last.left_padding, [2])
                self.assert_array(last.lengths, [5])
                joined = ArraysCache.merge([last, first])
                self.assertEqual(joined.batch_size, 2)
                self.assert_array(joined.left_padding, [2, 1])
                self.assert_array(joined.lengths, [5, 6])
                self.assert_array(joined.make_mask(3), [[False, False, True],
                                                       [False, True, True]])
                extended = last.extract(0)
                extended.extend(first)
                self.assert_array(extended.left_padding, [2, 1])
                self.assert_array(extended.lengths, [5, 6])
                if filled:
                    self.assert_array(joined[1], [[10, 11], [30, 31]])
                    self.assert_array(extended[1], [[10, 11], [30, 31]])
                    last[1][0, 0] = -1
                    self.assert_array(source[1], [[30, 31], [10, 11]])
                last.left_padding[0] = 99
                last.lengths[0] = 99
                self.assert_array(source.left_padding, [1, 2])
                self.assert_array(source.lengths, [6, 5])
                joined.advance(1)
                self.assert_array(joined.left_padding, [1, 0])
                self.assert_array(joined.lengths, [4, 5])
                joined.finalize()
                self.assertIsNone(joined.left_padding)
                self.assertIsNone(joined.lengths)
                for index in (-3, 2):
                    with self.assertRaises(IndexError):
                        source.extract(index)

    def test_arrays_guard_probe_lengths_only_and_cache_list(self):
        probe = ArraysCache(1)
        self.assertEqual(probe.extract(0).state, [None])
        self.assertEqual(probe.extract(-1).state, [None])
        lengths_only = ArraysCache(2)
        lengths_only.prepare(lengths=[3, 7])
        row = CacheList(lengths_only).extract(1)[0]
        self.assertIs(type(row), ArraysCache)
        self.assertIsNone(row.left_padding)
        self.assert_array(row.lengths, [7])
        joined = ArraysCache.merge([row, lengths_only.extract(0)])
        self.assertIsNone(joined.left_padding)
        self.assert_array(joined.make_mask(5), [[True] * 5,
                                              [True, True, True, False, False]])
        self.assertIsInstance(joined.state, list)
        self.assertEqual(len(joined.state), 2)

    def test_batch_kv_pending_padding_survives_reorder_and_join(self):
        cache = BatchKVCache([0, 0, 0])
        cache.prepare(lengths=[3, 1, 2], right_padding=[0, 2, 1])
        x = history([[11, 12, 13], [21, 0, 0], [31, 32, 0]])
        cache.update_and_fetch(x, x + 100)
        cache.filter([2, 0])
        self.assert_array(cache._right_padding, [1, 0])
        self.assertEqual(cache.batch_size, 2)
        self.assertFalse(cache.is_single_row())
        cache.finalize()
        self.assert_array(cache.offset, [2, 3])
        self.assert_array(cache.left_padding, [1, 0])
        rows = [cache.extract(0), cache.extract(1)]
        self.assert_history(rows[0], [31, 32])
        self.assert_history(rows[1], [11, 12, 13])
        joined = BatchKVCache.merge(rows[::-1])
        self.assert_history(joined.extract(0), [11, 12, 13])
        self.assert_history(joined.extract(1), [31, 32])
        joined.filter([1])
        self.assertTrue(joined.is_single_row())
        self.assertEqual(len(joined.state), 4)

    def test_rotating_final_single_token_prefill_and_filtered_lengths(self):
        cache = BatchRotatingKVCache(3, [0, 0, 0])
        cache.prepare(lengths=[4, 1, 2], right_padding=[0, 3, 2])
        x = history([[11, 12, 13], [21, 0, 0], [31, 32, 0]])
        cache.update_and_fetch(x, x + 100)
        cache.filter([2, 0])
        self.assert_array(cache._lengths, [2, 4])
        x = history([[0], [14]])
        cache.update_and_fetch(x, x + 100)
        cache.finalize()
        self.assert_array(cache.offset, [2, 4])
        self.assertIsNone(cache._lengths)
        self.assertEqual(cache.batch_size, 2)
        self.assertFalse(cache.is_single_row())
        row0, row1 = cache.extract(0), cache.extract(1)
        self.assert_array(row0.keys.reshape(-1), [31, 32])
        self.assert_array(row1.keys.reshape(-1), [12, 13, 14])
        self.assert_array(row0.values.reshape(-1), [131, 132])
        self.assert_array(row1.values.reshape(-1), [112, 113, 114])
        self.assertEqual((row0.offset, row1.offset), (2, 4))

    def test_rotating_merge_skips_allocated_but_zero_length_row(self):
        empty = RotatingKVCache(4)
        x = history([[99, 98]])
        empty.update_and_fetch(x, x + 100)
        empty.trim(2)
        full = RotatingKVCache(4)
        x = history([[7, 8, 9]])
        full.update_and_fetch(x, x + 100)
        merged = BatchRotatingKVCache.merge([empty, full, RotatingKVCache(4)])
        self.assert_array(merged.offset, [0, 3, 0])
        self.assert_array(merged.left_padding, [3, 0, 3])
        self.assert_array(merged.keys[:, 0, :, 0], [[0, 0, 0], [7, 8, 9], [0, 0, 0]])
        merged.filter([1])
        self.assertTrue(merged.is_single_row())
        self.assert_array(merged.extract(0).keys.reshape(-1), [7, 8, 9])

    def test_prefix_roundtrip_old_metadata_and_caller_detachment(self):
        for cache, fresh in [(KVCache(), KVCache()),
                             (RotatingKVCache(4), RotatingKVCache(4)),
                             (BatchKVCache([0]), BatchKVCache([0])),
                             (BatchRotatingKVCache(4, [0]),
                              BatchRotatingKVCache(4, [0]))]:
            with self.subTest(kind=type(cache).__name__):
                x = history([[1, 2]])
                cache.update_and_fetch(x, x + 100)
                # 1068's contract assigns detaching copies to the caller.
                snapshot = copy.deepcopy(cache.prefix_cache_snapshot())
                fresh.prefix_cache_restore(snapshot)
                self.assertEqual(fresh.meta_state, cache.meta_state)
                self.assertEqual(len(fresh.state), len(cache.state))
                self.assert_array(fresh.keys[:, :, :2, :].reshape(-1), [1, 2])
                cache.keys[0, 0, 0, 0] = 99
                self.assert_array(fresh.keys[:, :, :2, :].reshape(-1), [1, 2])
                self.assertIsNone(fresh.prefix_cache_merge([snapshot], [2]))
        rotating = BatchRotatingKVCache(2, [0])
        for token in (1, 2, 3):
            x = history([[token]])
            rotating.update_and_fetch(x, x + 100)
        restored = BatchRotatingKVCache(2, [0])
        restored.prefix_cache_restore(copy.deepcopy(rotating.prefix_cache_snapshot()))
        self.assertTrue(restored.rotated)
        self.assertEqual(restored.meta_state, rotating.meta_state)

    def test_quantized_apc_existing_affine_formats_and_metadata(self):
        for bits in (4, 8):
            with self.subTest(bits=bits):
                cache = QuantizedKVCache(group_size=32, bits=bits)
                self.assertEqual(cache.dequantize_for_apc(), (None, None))
                # Endpoints 0/1 and 2/3 are exactly representable in affine groups.
                keys = mx.tile(mx.array([0., 1.]), 48).reshape(1, 1, 3, 32)
                values = keys + 2
                cache.update_and_fetch(keys, values)
                cache.trim(1)
                snapshot = copy.deepcopy(cache.prefix_cache_snapshot())
                restored = QuantizedKVCache()
                restored.prefix_cache_restore(snapshot)
                self.assertEqual(restored.meta_state, ('2', '32', str(bits)))
                self.assertEqual(len(restored.state), 2)
                k, v = restored.dequantize_for_apc()
                self.assertEqual(k.shape, (1, 1, 2, 32))
                self.assertTrue(mx.allclose(k, keys[:, :, :2]).item())
                self.assertTrue(mx.allclose(v, values[:, :, :2]).item())
                restored.trim(2)
                self.assertEqual(restored.dequantize_for_apc(), (None, None))


if __name__ == '__main__':
    unittest.main()
