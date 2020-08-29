import numpy as np

def get_continous_time_periods(binary_array):
    """
    take a binary array and return a list of tuples representing the first and last position(included) of continuous
    positive period
    This code was copied from another project or from a forum, but i've lost the reference.
    :param binary_array:
    :return:
    """
    binary_array = np.copy(binary_array).astype("int8")
    # first we make sure it's binary
    if np.max(binary_array) > 1:
        binary_array[binary_array > 1] = 1
    if np.min(binary_array) < 0:
        binary_array[binary_array < 0] = 0
    n_times = len(binary_array)
    d_times = np.diff(binary_array)
    # show the +1 and -1 edges
    pos = np.where(d_times == 1)[0] + 1
    neg = np.where(d_times == -1)[0] + 1

    if (pos.size == 0) and (neg.size == 0):
        if len(np.nonzero(binary_array)[0]) > 0:
            return [(0, n_times-1)]
        else:
            return []
    elif pos.size == 0:
        # i.e., starts on an spike, then stops
        return [(0, neg[0])]
    elif neg.size == 0:
        # starts, then ends on a spike.
        return [(pos[0], n_times-1)]
    else:
        if pos[0] > neg[0]:
            # we start with a spike
            pos = np.insert(pos, 0, 0)
        if neg[-1] < pos[-1]:
            #  we end with aspike
            neg = np.append(neg, n_times - 1)
        # NOTE: by this time, length(pos)==length(neg), necessarily
        # h = np.matrix([pos, neg])
        h = np.zeros((2, len(pos)), dtype="int16")
        h[0] = pos
        h[1] = neg
        if np.any(h):
            result = []
            for i in np.arange(h.shape[1]):
                if h[1, i] == n_times-1:
                    result.append((h[0, i], h[1, i]))
                else:
                    result.append((h[0, i], h[1, i]-1))
            return result
    return []


def get_unit_label(cluster_label, cluster_index, channel_index, region_label):
    """
    Return the string representing an unit
    :param cluster_label: (str) such as 'SU', 'MU'
    :param cluster_index: (int) representing the cluster, allows to distinguish SU & MU
    :param channel_index: (int) micro_wire index, the one present in 'times_pos_CSC2.mat'
    :param region_label: (str) such as LMH5 for Left Medial Hippocampus 5 (5 being the wire)
    :return:
    """
    return f"{cluster_label} {cluster_index} {channel_index} {region_label}"


def get_brain_area_from_cell_label(cell_label):
    """
    Get the brain area string from the label of a cell
    :param cell_label: (str) such as 'MU 7 40 RA1' or 'SU 1 67 RMH4'
    :return: None if the format is not valid, a str otherwise
    """
    split_values = cell_label.split()
    if len(split_values) != 4:
        return None
    return split_values[-1][1:-1]
