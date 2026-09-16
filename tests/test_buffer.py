from terra.utils.buffer import MultiRingBuffer
import numpy as np


def test_buffer():
    max_len = 20
    check_1 = 10
    check_2 = 30
    seq_len = 5
    B = 8
    num_checks = 100

    keys = ["key1", "key2"]
    dims = [1, 2]
    example_item = {keys[i]: np.zeros((dims[i],)) for i in range(len(keys))}

    buf = MultiRingBuffer(max_len, example_item)
    i = 0

    for _ in range(check_1):
        item = {k: np.ones((d,)) * i for k, d in zip(keys, dims)}
        buf.append(item)
        i += 1

    batch = buf.sample_sequences(B, seq_len)
    for k, d in zip(keys, dims):
        print(f"Key {k} batch.shape:\n{batch[k].shape}")
        assert batch[k].shape == (B, seq_len, d)
        print(f"Key {k} batch, dim 0:\n{batch[k][:, :, 0]}")

    for _ in range(num_checks):
        batch = buf.sample_sequences(B, seq_len)
        _check_batch(batch, keys)

    for _ in range(check_2):
        item = {k: np.ones((d,)) * i for k, d in zip(keys, dims)}
        buf.append(item)
        i += 1

    for _ in range(num_checks):
        batch = buf.sample_sequences(B, seq_len)
        _check_batch(batch, keys)

    batch = buf.sample_sequences(B, seq_len, include_latest=3)
    print("Batch with include_latest=3, dim 0:")
    for k in keys:
        print(f"Key {k} batch, dim 0:\n{batch[k][:, :, 0]}")


def test_episode_boundaries():
    print("Running episode boundary test...")
    max_len = 100
    seq_len = 5
    B = 10

    example_item = {"obs": np.zeros((1,))}
    buf = MultiRingBuffer(max_len, example_item)

    # --- Episode 1 ---
    # Values: 0 to 9
    # This episode is clean (no internal resets)
    for i in range(10):
        # First item is technically an episode start, but it's at index 0 so it doesn't matter much
        is_start = i == 0
        buf.append({"obs": np.array([float(i)])}, episode_start=is_start)

    # --- Episode 2 ---
    # Values: 100 to 109
    # We use a large jump in values (9 -> 100) to easily detect if a sequence crosses the boundary.
    # The item '100' is a RESET.
    for i in range(10):
        val = 100 + i
        is_start = i == 0  # This is the crucial flag
        buf.append({"obs": np.array([float(val)])}, episode_start=is_start)

    # Sample heavily to try and catch an error
    for _ in range(50):
        batch = buf.sample_sequences(B, seq_len)
        data = batch["obs"][:, :, 0]  # (B, T)

        # Check every sequence in the batch
        for b in range(B):
            seq = data[b]

            # Logic check 1: A sequence should not contain both 'Low values' (Ep1) and 'High values' (Ep2)
            has_ep1 = np.any(seq < 50)
            has_ep2 = np.any(seq >= 100)

            if has_ep1 and has_ep2:
                print("FAILED: Sequence crossed episode boundary!")
                print(f"Sequence: {seq}")
                assert False, "Found a sequence merging two episodes."

            # Logic check 2: Strict increment check
            # Within an episode, our data strictly increments by 1.
            # If we crossed a boundary, we would see a jump (e.g. 9 -> 100)
            diffs = np.diff(seq)
            if not np.allclose(diffs, 1.0):
                print("FAILED: Sequence discontinuity detected!")
                print(f"Sequence: {seq}")
                print(f"Diffs: {diffs}")
                assert False, "Sequence not contiguous (likely crossed boundary)."

    print("Episode boundary test passed.")


def _check_batch(batch: dict[str, np.ndarray], keys: list[str]):
    """
    Verify that:
      1. Each sequence is contiguous in time (increments by 1 along dim 0).
      2. All keys are aligned (same underlying indices in dim 0).
    """
    # (1) Contiguity check: each row in batch[k][:, :, 0] should increase by 1
    for k in keys:
        diffs = np.diff(batch[k][:, :, 0], axis=1)
        assert np.all(
            diffs == 1
        ), f"Non-contiguous sequence detected for key {k}: diffs=\n{diffs}"

    # (2) Alignment check: compare the "time indices" across all keys
    ref = batch[keys[0]][:, :, 0]
    for k in keys[1:]:
        assert np.array_equal(
            ref, batch[k][:, :, 0]
        ), f"Keys not aligned in time for key {k}"
