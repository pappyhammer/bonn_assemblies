"""
Spike trains utils
"""
import numpy as np
from bisect  import bisect      # for binary search in sorted lists
from random  import random, randint, sample, choice
import numpy as np
import math
from pymeica.utils.misc import get_continous_time_periods
import elephant.conversion as elephant_conv
import neo
import quantities as pq


def get_spike_times_in_bins(units, spike_indices, bins_to_explore, spike_trains):
    """
    For each unit, get the spike times corresponding to the bins given
    Args:
        units: int representing the index of the unit
        spike_indices: A list of lists for each spike train (i.e., rows of the binned matrix),
        that in turn contains for each spike the index into the binned matrix where this spike enters.
        bins_to_explore: array of int representing the bin to explore
        spike_trains: list of lists for each spike train, containing the timestamps of spikes (non binned)

    Returns: a list of units and spike times, of the same lengths, the units indices corresponding to the spike times

    """
    units_index_by_spike = []
    spike_times = []
    # same number of units, but we only keep the spike times that are in those bins
    new_spike_trains = [[]]*len(units)

    for bin_number in bins_to_explore:
        # print(f"bin_number {bin_number}")
        for unit_index, unit_id in enumerate(units):
            # print(f"unit_index {unit_index}, unit_id {unit_id}")
            # print(f"spike_nums[unit_id][bin_number] {spike_nums[unit_id][bin_number]}")
            unit_spike_indices = spike_indices[unit_id]
            # print(f"unit_spike_indices {unit_spike_indices}")
            spikes_in_bin = np.where(unit_spike_indices == bin_number)[0]
            if len(spikes_in_bin) > 0:
                # print(f"spikes_in_bin {spikes_in_bin}")
                for spike_in_bin in spikes_in_bin:
                    spike_times.append(spike_trains[unit_id][spike_in_bin])
                    units_index_by_spike.append(unit_index)
                    new_spike_trains[unit_index].append(spike_trains[unit_id][spike_in_bin])

    return units_index_by_spike, spike_times, new_spike_trains

# TODO: same method but with spike_trains
# TODO: for concatenation of SCE, if the same cells spike more than one, then the following should be considered
# in the count of cells active after the first SCE
def detect_sce_with_sliding_window(spike_nums, window_duration, perc_threshold=95,
                                   with_refractory_period=-1, non_binary=False,
                                   activity_threshold=None, debug_mode=False,
                                   no_redundancy=False, keep_only_the_peak=False):

    """
    Use a sliding window to detect sce (define as peak of activity > perc_threshold percentile after
    randomisation during a time corresponding to window_duration)
    :param spike_nums: 2D array, lines=cells, columns=time
    :param window_duration:
    :param perc_threshold:
    :param no_redundancy: if True, then when using the sliding window, a second spike of a cell is not taking into
    consideration when looking for a new SCE
    :param keep_only_the_peak: keep only the frame with the maximum cells co-activating
    :return: ** one array (mask, boolean) containing True for indices (times) part of an SCE,
    ** a list of tuple corresponding to the first and last index of each SCE, (last index being included in the SCE)
    ** sce_nums: a new spike_nums with in x axis the SCE and in y axis the neurons, with 1 if
    active during a given SCE.
    ** an array of len n_times, that for each times give the SCE number or -1 if part of no cluster
    ** activity_threshold

    """

    if non_binary:
        binary_spikes = np.zeros((len(spike_nums), len(spike_nums[0, :])), dtype="int8")
        for neuron, spikes in enumerate(spike_nums):
            binary_spikes[neuron, spikes > 0] = 1
        spike_nums = binary_spikes

    if activity_threshold is None:
        activity_threshold = get_sce_detection_threshold(spike_nums=spike_nums, n_surrogate=1000,
                                                         window_duration=window_duration,
                                                         perc_threshold=perc_threshold,
                                                         non_binary=False)

    n_cells = len(spike_nums)
    n_times = len(spike_nums[0, :])

    if window_duration == 1:
        # using a diff method
        sum_spike_nums = np.sum(spike_nums, axis=0)
        binary_sum = np.zeros(n_times, dtype="int8")
        binary_sum[sum_spike_nums >= activity_threshold] = 1
        sce_tuples = get_continous_time_periods(binary_sum)
        if keep_only_the_peak:
            new_sce_tuples = []
            for sce_index, sce_tuple in enumerate(sce_tuples):
                index_max = np.argmax(np.sum(spike_nums[:, sce_tuple[0]:sce_tuple[1] + 1], axis=0))
                new_sce_tuples.append((sce_tuple[0]+index_max, sce_tuple[0]+index_max))
            sce_tuples = new_sce_tuples
        sce_bool = np.zeros(n_times, dtype="bool")
        sce_times_numbers = np.ones(n_times, dtype="int16")
        sce_times_numbers *= -1
        for sce_index, sce_tuple in enumerate(sce_tuples):
            sce_bool[sce_tuple[0]:sce_tuple[1]+1] = True
            sce_times_numbers[sce_tuple[0]:sce_tuple[1]+1] = sce_index

    else:
        start_sce = -1
        # keep a trace of which cells have been added to an SCE
        cells_in_sce_so_far = np.zeros(n_cells, dtype="bool")
        sce_bool = np.zeros(n_times, dtype="bool")
        sce_tuples = []
        sce_times_numbers = np.ones(n_times, dtype="int16")
        sce_times_numbers *= -1
        if debug_mode:
            print(f"n_times {n_times}")
        for t in np.arange(0, (n_times - window_duration)):
            if debug_mode:
                if t % 10**6 == 0:
                    print(f"t {t}")
            cells_has_been_removed_due_to_redundancy = False
            sum_value_test = np.sum(spike_nums[:, t:(t + window_duration)])
            sum_spikes = np.sum(spike_nums[:, t:(t + window_duration)], axis = 1)
            pos_cells = np.where(sum_spikes)[0]
            # neurons with sum > 1 are active during a SCE
            sum_value = len(pos_cells)
            if no_redundancy and (start_sce > -1):
                # removing from the count the cell that are in the previous SCE
                nb_cells_already_in_sce = np.sum(cells_in_sce_so_far[pos_cells])
                sum_value -= nb_cells_already_in_sce
                if nb_cells_already_in_sce > 0:
                    cells_has_been_removed_due_to_redundancy = True
            # print(f"Sum value, test {sum_value_test}, rxeal {sum_value}")
            if sum_value >= activity_threshold:
                if start_sce == -1:
                    start_sce = t
                    if no_redundancy:
                        # keeping only cells spiking at time t, as we're gonna shift of one on the next step
                        sum_spikes = np.sum(spike_nums[:, t])
                        pos_cells = np.where(sum_spikes)[0]
                        cells_in_sce_so_far[pos_cells] = True
                else:
                    if no_redundancy:
                        # updating which cells are already in the SCE
                        # keeping only cells spiking at time t, as we're gonna shift of one on the next step
                        sum_spikes = np.sum(spike_nums[:, t])
                        pos_cells = np.where(sum_spikes)[0]
                        cells_in_sce_so_far[pos_cells] = True
                    else:
                        pass
            else:
                if start_sce > -1:
                    if keep_only_the_peak:
                        index_max = np.argmax(spike_nums[:, start_sce:(t + window_duration) - 1])
                        sce_tuples.append((sce_tuple[0] + index_max, sce_tuple[0] + index_max))
                        sce_bool[sce_tuple[0] + index_max] = True
                        # sce_tuples.append((start_sce, t-1))
                        sce_times_numbers[sce_tuple[0] + index_max] = len(sce_tuples) - 1
                    else:
                        # then a new SCE is detected
                        sce_bool[start_sce:(t + window_duration) - 1] = True
                        sce_tuples.append((start_sce, (t + window_duration) - 2))
                        # sce_tuples.append((start_sce, t-1))
                        sce_times_numbers[start_sce:(t + window_duration) - 1] = len(sce_tuples) - 1

                    start_sce = -1
                    cells_in_sce_so_far = np.zeros(n_cells, dtype="bool")
                if no_redundancy and cells_has_been_removed_due_to_redundancy:
                    sum_value += nb_cells_already_in_sce
                    if sum_value >= activity_threshold:
                        # then a new SCE start right after the old one
                        start_sce = t
                        cells_in_sce_so_far = np.zeros(n_cells, dtype="bool")
                        if no_redundancy:
                            # keeping only cells spiking at time t, as we're gonna shift of one on the next step
                            sum_spikes = np.sum(spike_nums[:, t])
                            pos_cells = np.where(sum_spikes)[0]
                            cells_in_sce_so_far[pos_cells] = True

    n_sces = len(sce_tuples)
    sce_nums = np.zeros((n_cells, n_sces), dtype="int16")
    for sce_index, sce_tuple in enumerate(sce_tuples):
        sum_spikes = np.sum(spike_nums[:, sce_tuple[0]:(sce_tuple[1] + 1)], axis=1)
        # neurons with sum > 1 are active during a SCE
        active_cells = np.where(sum_spikes)[0]
        sce_nums[active_cells, sce_index] = 1

    # print(f"number of sce {len(sce_tuples)}")

    return sce_bool, sce_tuples, sce_nums, sce_times_numbers, activity_threshold


def create_spike_train_neo_format(spike_trains, time_format="sec"):
    """
    Take a spike train in sec and prepare it to be transform in neo format.
    Args:
        spike_trains: list of list or list of np.array, representing for each cell the timestamps in sec of its spikes
        time_format: (str) time format, could be "ms" or "sec", default is "sec"

    Returns: a new spike_trains in ms and the first and last timestamp (chronologically) of the spike_train.

    """

    new_spike_trains = []
    t_start = None
    t_stop = None
    for cell in np.arange(len(spike_trains)):
        spike_train = spike_trains[cell]
        if time_format == "ms":
            pass
        else:
            # then time_format is considered being sec
            # convert frames in ms
            spike_train = spike_train / 1000
        new_spike_trains.append(spike_train)
        if t_start is None:
            t_start = spike_train[0]
        else:
            t_start = min(t_start, spike_train[0])
        if t_stop is None:
            t_stop = spike_train[-1]
        else:
            t_stop = max(t_stop, spike_train[-1])

    return new_spike_trains, t_start, t_stop


def create_binned_spike_train(spike_trains, spike_trains_binsize, time_format="sec"):
    # first we create a spike_trains in the neo format
    spike_trains, t_start, t_stop = create_spike_train_neo_format(spike_trains, time_format=time_format)

    duration_in_sec = (t_stop - t_start) / 1000
    n_cells = len(spike_trains)

    neo_spike_trains = []

    for cell in np.arange(n_cells):
        spike_train = spike_trains[cell]
        # print(f"n_spikes: {cells_label[cell]}: {len(spike_train)}")
        neo_spike_train = neo.SpikeTrain(times=spike_train, units='ms',
                                         t_start=t_start,
                                         t_stop=t_stop)
        neo_spike_trains.append(neo_spike_train)

    spike_trains_binned = elephant_conv.BinnedSpikeTrain(neo_spike_trains, binsize=spike_trains_binsize)

    spike_nums = spike_trains_binned.to_bool_array().astype("int8")

    # A list of lists for each spike train (i.e., rows of the binned matrix),
    # that in turn contains for each spike the index into the binned matrix where this spike enters.
    spike_bins_indices = spike_trains_binned.spike_indices

    return spike_nums, spike_bins_indices



def spike_trains_threshold_by_firing_rate(spike_trains, firing_rate_threshold, duration_in_sec):
    """
    Remove cells that fire the most
    Args:
        spike_trains:
        firing_rate_threshold: (float) firing rate (number of spikes by second) above which we don't keep the cell.
        duration_in_sec: (float) total duration in sec to use to measure the firing rate. The same duration is used
        for each

    Returns: a new spike trains (list of list) without the cells firing above the threshold, and the list of cell
    indices kept

    """

    filtered_spike_trains = []
    cells_below_threshold = []
    for cell in np.arange(len(spike_trains)):
        spike_train = spike_trains[cell]
        n_spike_normalized = len(spike_train) / duration_in_sec
        # print(f"n spikes: {n_spike_normalized}")
        if n_spike_normalized <= firing_rate_threshold:
            filtered_spike_trains.append(spike_train)
            cells_below_threshold.append(cell)

    return filtered_spike_trains, cells_below_threshold

def get_sce_detection_threshold(spike_nums, window_duration, n_surrogate, use_max_of_each_surrogate=False,
                                perc_threshold=95, non_binary=False,
                                debug_mode=False, spike_train_mode=False):
    """
    Compute the activity threshold (ie: nb of onset at each time, if param.bin_size > 1, then will first bin
    the spike by bin_size times then compute the threshold.
    :param spike_nums: (2d np.array n_cells * timepoint), at each time point, indicate if the cell has one or more
    spikes (could be binary 1 if at least a spike, or the number of spikes)
    :param non_binary: means that spike_nums could hold values that are not only 0 or 1
    :param spike_train_mode: if True, spike_nums should be a list of np.array with float or int value
    representing the spike time of each cell (each np.array representing a cell)
    :param use_max_of_each_surrogate: if True, the percentile threshold will be applied to the max sum of each
    surrogate generated.
    :return:
    """
    if debug_mode:
        print("start get activity threshold")

    if spike_train_mode:
        min_time, max_time = trains_module.get_range_train_list(spike_nums)
        surrogate_data_set = create_surrogate_dataset(train_list=spike_nums, nsurrogate=n_surrogate,
                                                      min_value=min_time, max_value=max_time)
        n_times = int(math.ceil(max_time - min_time))
        n_cells = len(spike_nums)
        just_keeping_the_max_of_each_surrogate = use_max_of_each_surrogate

        number_of_times_without_spikes = 0

        if just_keeping_the_max_of_each_surrogate:
            n_rand_sum = []
        else:
            n_rand_sum = np.zeros(0)

        if debug_mode:
            print(f"window_duration {window_duration}")
        for i, surrogate_train_list in enumerate(surrogate_data_set):
            if debug_mode:
                if (i % 5) == 0:
                    print(f"surrogate n°: {i}")
            # to make it faster, we keep the count of cells in a dict, thus not having to create a huge
            # matrix if only a sparse number of times have spikes
            # this dict will have as key the cell number and as value a set containing
            # the time in wich a spike was counting as part of active during a window
            # using a set allows to keep it simple and save computational time (hopefully)
            windows_set = dict()
            for cell_number in np.arange(n_cells):
                windows_set[cell_number] = set()

            for cell, spikes_train in enumerate(surrogate_train_list):
                # if debug_mode and (cell == 0):
                #     print(f"len(spikes_train): {len(spikes_train)}")
                for spike_time in spikes_train:
                    # first determining to which windows to add the spike
                    spike_index = int(spike_time - min_time)
                    first_index_window = np.max((0, int(spike_index - window_duration)))
                    # we add to the set of the cell, all indices in this window
                    windows_set[cell].update(np.arange(first_index_window, spike_index))

            # uint8 : int from 0 to 255
            # max sum should be n_cells
            # for memory optimization
            if n_cells < 255:
                count_array = np.zeros(n_times, dtype="uint8")
            else:
                count_array = np.zeros(n_times, dtype="uint16")
            for cell, times in windows_set.items():
                times = np.asarray(list(times))
                # mask = np.zeros(n_times, dtype="bool")
                # mask[times] = True
                count_array[times] = count_array[times] + 1

            # print("after windows_sum")
            sum_spikes = count_array[count_array>0]
            # not to have to keep a huge array, we just keep values superior to 0 and we keep the count
            # off how many times are at 0
            number_of_times_without_spikes += (n_times - (len(count_array) - len(sum_spikes)))
            # concatenating the sum of spikes for each time
            if just_keeping_the_max_of_each_surrogate:
                n_rand_sum.append(np.max(sum_spikes))
            else:
                n_rand_sum = np.concatenate((n_rand_sum, sum_spikes))

        if just_keeping_the_max_of_each_surrogate:
            n_rand_sum = np.asarray(n_rand_sum)
        else:
            # if debug_mode:
            #     print(f"number_of_times_without_spikes {number_of_times_without_spikes}")
            n_rand_sum = np.concatenate((n_rand_sum, np.zeros(number_of_times_without_spikes, dtype="uint16")))
            pass

        activity_threshold = np.percentile(n_rand_sum, perc_threshold)

        return activity_threshold

    # ------------------- for non spike_train_mode ------------------

    if non_binary:
        binary_spikes = np.zeros((len(spike_nums), len(spike_nums[0, :])), dtype="int8")
        for neuron, spikes in enumerate(spike_nums):
            binary_spikes[neuron, spikes > 0] = 1
        spike_nums = binary_spikes

    n_times = len(spike_nums[0, :])

    # computing threshold to detect synchronous peak of activity
    if use_max_of_each_surrogate:
        n_rand_sum = np.zeros(n_surrogate)
    else:
        if window_duration == 1:
            n_rand_sum = np.zeros(n_surrogate * n_times )
        else:
            n_rand_sum = np.zeros(n_surrogate * (n_times-window_duration))
    for i in np.arange(n_surrogate):
        if debug_mode:
            print(f"surrogate n°: {i}")
        copy_spike_nums = np.copy(spike_nums)
        for n, neuron_spikes in enumerate(copy_spike_nums):
            # roll the data to a random displace number
            copy_spike_nums[n, :] = np.roll(neuron_spikes, np.random.randint(1, n_times))
        if window_duration == 1:
            n_rand_sum[i*n_times:(i+1)*n_times] = np.sum(copy_spike_nums, axis=0)
            continue
        max_sum = 0
        for t in np.arange(0, (n_times - window_duration)):
            sum_value = np.sum(copy_spike_nums[:, t:(t + window_duration)])
            max_sum = np.max((sum_value, max_sum))
            if not use_max_of_each_surrogate:
                n_rand_sum[(i * (n_times - window_duration)) + t] = sum_value

        # for t in np.arange((n_times - window_duration), n_times):
        #     sum_value = np.sum(spike_nums[:, t:])
        #     n_rand_sum[(i * n_times) + t] = sum_value

        # Keeping the max value for each surrogate data
        if use_max_of_each_surrogate:
            n_rand_sum[i] = max_sum

    activity_threshold = np.percentile(n_rand_sum, perc_threshold)

    return activity_threshold

def create_surrogate_dataset(train_list, nsurrogate, min_value, max_value):
    """

    :param train_list:
    :param nsurrogate:
    :param sigma: noise of jittering
    :return:
    """
    surrogate_data_set = []
    for i in range(nsurrogate):
        surrogate_data_set.append(dithered_data_set(train_list, min_value, max_value))
    return surrogate_data_set


def dithered_data_set(train_list, min_value, max_value):
    """
        create new train list with jittered spike trains
    :param train_list: the spike train list to be jittered
    :param sigma: noise of jittering
    :return: new jittered train list

    """
    jittered_list = []
    for train in train_list:
        new_train = trains_module.shifted(train, shift=(max_value-min_value), rng=(min_value, max_value))
        jittered_list.append(new_train)
    return jittered_list

#-----------------------------------------------------------------------
# File    : trains.py
# Contents: spike train modification functions
# Authors : Sebastien Louis, Christian Borgelt
# History : 2009.08.?? file created
#           2009.08.20 spike train dithering improved
#           2009.08.24 spike shifting improved (general ranges)
#           2009.08.25 interspike interval dithering added
#           2009.09.02 changes for compatibility with Python 2.5.2
#-----------------------------------------------------------------------


#-----------------------------------------------------------------------
# Spike Train Functions
#-----------------------------------------------------------------------

def randomized (train, rng=range(1000)):
    '''Create a randomized (binned) spike train
    (that is, assign random new spike time bin indices).
    train: spike train to randomize
    rng:   range of allowed time bin indices'''
    return sample(rng, len(train))

# -----------------------------------------------------------------------


def from_spike_trains_to_spike_nums(spike_trains):
    min_time, max_time = get_range_train_list(spike_trains)
    n_times = int(np.ceil((max_time-min_time)+1))
    spike_nums = np.zeros((len(spike_trains), n_times), dtype="int16")
    for train_index, train in enumerate(spike_trains):
        # normalizing
        train = train - min_time
        # rounding to integers
        train = train.astype(int)
        # mask = np.zeros(n_times, dtype="bool")
        # mask[train] = True
        spike_nums[train_index, train] = 1
    return spike_nums

def from_spike_nums_to_spike_trains(spike_nums):
    spike_trains = []
    for spikes in spike_nums:
        spike_trains.append(np.where(spikes)[0])
    return spike_trains

def get_range_train_list(train_list):
    """

    :param train_list:
    :return: the min and max value of the train list
    """
    min_value = 0
    max_value = 0

    for i, train in enumerate(train_list):
        if len(train) == 0:
            return
        if i == 0:
            min_value = np.min(train)
            max_value = np.max(train)
            continue
        min_value = min(min_value, np.min(train))
        max_value = max(max_value, np.max(train))

    return min_value, max_value


def shifted (train, shift=20, rng=range(1000)):
    '''Create a randomly shifted/rotated (binned) spike train.
    train: spike train to shift
    shift: maximum amount by which to shift the spikes
    rng:   range of allowed time bin indices
    returns a spike trains with the same number of spikes'''
    off = rng[0]
    n = rng[-1] + 1 - off
    shift = int(abs(shift)) % n  # compute the shift value
    shift = randint(-shift, shift) + n - off
    return [((x+shift) % n) + off for x in train]

#-----------------------------------------------------------------------

def dithered_fast (train, dither=20, rng=range(1000)):
    '''Create a dithered (binned) spike train
    (that is, modify spike time bin indices).
    train:  spike train to dither
    dither: maximum amount by which to shift a spike
    rng:    range of allowed time bin indices
    returns a spike train with dithered spike times'''
    d = [randint(max(x-dither,rng[ 0]),
                 min(x+dither,rng[-1])) for x in train]
    return [x for x in set(d)]  # remove possible duplicates

#-----------------------------------------------------------------------

def dithered (train, dither=20, rng=range(1000)):
    '''Create a dithered (binned) spike train
    (that is, modify spike time bin indices).
    train:  spike train to dither
    dither: maximum amount by which to shift a spike
    rng:    range of allowed time bin indices
    returns a spike train with dithered spike times'''
    d = [randint(max(x-dither,rng[ 0]),
                 min(x+dither,rng[-1])) for x in train]
    # This way of writing the dithering (randint call) ensures that
    # all new time bin indices lie in the allowed time bin index range.
    # The execution speed penalty for this is relatively small.
    if len(set(d)) == len(d): return d
    s = set()                   # check wether all bin indices differ
    for i in range(len(d)):    # if not, re-dither the duplicates
        if d[i] not in s: s.add(d[i]); continue
        r = range(max(rng[ 0],train[i]-dither),
                   min(rng[-1],train[i]+dither))
        r = [x for x in r if x not in s]
        if r: d[i] = choice(r); s.add(d[i])
    return d if len(d) == len(s) else [x for x in s]
    # Simply returning the initially created d may lose spikes,
    # because the initial d may contain duplicate bin indices.
    # This version tries to maintain the spike count if possible.
    # Only if all bins in the jitter window around some spike are
    # already taken, the spike cannot be dithered and is dropped.

#-----------------------------------------------------------------------

def isi_dithered (train, cdfs, isid=20, rng=range(1000)):
    '''Create a dithered (binned) spike train
    (that is, modify spike time bin indices).
    train:  spike train to dither
    cdfs:   cumulative distribution functions for the
            interspike interval pairs with the same sum
    isid:   maximum amount by which to shift a spike
    rng:    range of allowed time bin indices
    returns a spike train with dithered spike times'''
    train = [rng[0]-1] +train +[rng[-1]+1]
    size  = len(cdfs)           # expand train by dummy spikes and
    out   = set()               # initialize the output spike set
    for i in range(1,len(train)-1):
        a,s,b = train[i-1],train[i],train[i+1]
        x = s-a; k = x+(b-s)-2  # get three consecutive spikes
        if k < size:            # if inside range of distributions
            d = cdfs[k]         # dither according to isi distribution
            x = max(0,x-isid-1); y = min(len(d)-1,x+isid)
            for j in range(isid+isid):
                k = a +bisect(d[x:y],d[x]+random()*(d[y]-d[x]))
                if k not in out: out.add(k); break
        else:                   # if outside range of distributions
            r = range(max(a+1,s-isid),min(b-1,s+isid))
            r = [x for x in r if x not in out]
            if r: out.add(choice(r)) # dither with uniform distribution
    return [s for s in out]     # return the dithered spike train
