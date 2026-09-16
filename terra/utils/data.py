import numpy as np


def generate_valid_indices(num_episodes, episode_length, beg_clip, end_clip):
    """
    Generate valid indices for a given number of episodes and episode length.

    Args:
        num_episodes (int): Number of episodes.
        episode_length (int): Length of each episode.
        beg_clip (int): Number of beginning steps to clip off per episode.
        end_clip (int): Number of end steps to clip off per episode.

    Returns:
        np.ndarray: Valid indices for the dataset.
    """
    if beg_clip < 0 or end_clip < 0:
        raise ValueError("beg_clip and end_clip must be non-negative integers.")

    if episode_length <= beg_clip + end_clip:
        raise ValueError("Episode length must be greater than beg_clip + end_clip.")

    # Calculate valid indices
    episode_starts = np.arange(num_episodes) * episode_length
    valid_in_episode = np.arange(beg_clip, episode_length - end_clip)
    valid_indices = (episode_starts[:, None] + valid_in_episode[None, :]).reshape(-1)

    return valid_indices
