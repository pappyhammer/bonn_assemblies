import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import patches
import seaborn as sns
from datetime import datetime
import matplotlib.cm as cm
import numpy as np
import math
import os
from cicada.utils.signal import gaussblur1D, norm01
from cicada.utils.misc import bin_raster


def plot_raster(spike_nums=None, title=None, file_name=None,
                spike_train_format=False,
                y_ticks_labels=None,
                y_ticks_labels_size=None,
                y_ticks_labels_color="white",
                hide_raster_y_ticks_labels=True,
                x_ticks_labels_color="white",
                x_ticks_labels=None,
                x_ticks_labels_size=None,
                x_ticks=None,
                hide_x_ticks_labels=False,
                figure_background_color="black",
                without_ticks=True,
                save_raster=False,
                show_raster=False,
                plot_with_amplitude=False,
                activity_threshold=None,
                save_formats="png",
                span_area_coords=None,
                span_area_colors=None,
                span_area_only_on_raster=True,
                alpha_span_area=0.5,
                cells_to_highlight=None,
                cells_to_highlight_colors=None,
                color_peaks_activity=False,
                horizontal_lines=None,
                horizontal_lines_colors=None,
                horizontal_lines_sytle=None,
                horizontal_lines_linewidth=None,
                vertical_lines=None,
                vertical_lines_colors=None,
                vertical_lines_sytle=None,
                vertical_lines_linewidth=None,
                scatters_on_traces=None,
                scatters_on_traces_marker="*",
                scatters_on_traces_size=5,
                sliding_window_duration=1,
                show_sum_spikes_as_percentage=False,
                span_cells_to_highlight=None,
                span_cells_to_highlight_colors=None,
                spike_shape="|",
                spike_shape_size=10,
                raster_face_color='black',
                cell_spikes_color='white',
                activity_sum_plot_color="white",
                activity_sum_face_color="black",
                y_lim_sum_activity=None,
                seq_times_to_color_dict=None,
                link_seq_categories=None,
                link_seq_color=None, min_len_links_seq=3,
                link_seq_line_width=1, link_seq_alpha=1,
                jitter_links_range=1,
                display_link_features=True,
                seq_colors=None, debug_mode=False,
                axes_list=None,
                SCE_times=None,
                raster_y_axis_label=None,
                raster_y_axis_label_size=None,
                axes_label_color='white',
                without_activity_sum=False,
                spike_nums_for_activity_sum=None,
                spikes_sum_to_use=None,
                size_fig=None,
                cmap_name="jet", traces=None,
                display_traces=False,
                display_spike_nums=True,
                traces_lw=0.3,
                path_results=None,
                xkcd_mode=False,
                desaturate_color_according_to_normalized_amplitude=False,
                lines_to_display=None,
                lines_color="white",
                lines_width=1,
                lines_band=0,
                lines_band_color="white",
                use_brewer_colors_for_traces=False,
                dpi=100,
                with_timestamp_in_file_name=True,
                display_dashed_line_with_traces=False
                ):
    """
    Plot or save a raster given a 2d array either binary representing onsets, peaks or rising time, or made of float
    to represents traces or encoding in onset/peaks/rising time a value.
    :param spike_nums: np.array of 2D, axis=1 (lines) represents the cells, the columns representing the spikes
    It could be binary, or containing the amplitude, if amplitudes values should be display put plot_with_amplitude
    to True
    :param spike_train_format: if True, means the data is a list of np.array, and then spike_nums[i][j] is
    a timestamps value as float
    :param title: title to be plot
    :param file_name: name of the file if save_raster is True
    :param save_raster: if True, the plot will be save. To do so param should not be None and contain a variable
    path_results that will indicated where to save file_name
    :param show_raster: if True, the plot will be shown
    :param plot_with_amplitude: to display a color bar representing the content values.
    :param activity_threshold: Int representing a threshold that will be display as a red line on the sum of activity
    subplot.
    :param save_formats: string or list of string representing the formats in which saving the raster.
    Exemple: "pdf" or ["pdf", "png"]
    :param span_area_coords: List of list of tuples of two float representing coords (x, x) of span band with a color
    corresponding to the one in span_area_colors
    :param span_area_colors: list of colors, same len as span_area_coords
    :param span_area_only_on_raster: if True, means the span won't be on the sum of activity on the sub-plot as well
    :param cells_to_highlight: cells index to span (y-axis) with special spikes color, list of int
    :param cells_to_highlight_colors: cells colors to span, same len as cells_to_span, list of string
    :param color_peaks_activity: if True, will span to the color of cells_to_highlight_colors each time at which a cell
    among cells_to_highlight will spike on the activity peak diagram
    :param horizontal_lines: list of float, representing the y coord at which trace horizontal lines
    :param horizontal_lines_colors: if horizontal_lines is not None, will set the colors of each line,
    list of string or color code
    :param horizontal_lines_style: give the style of the lines, string
    :param vertical_lines: list of float, representing the x coord at which trace vertical lines
    :param vertical__lines_colors: if horizontal_lines is not None, will set the colors of each line,
    list of string or color code
    :param vertical__lines_style: give the style of the lines, string
    :param vertical_lines_linewidth: linewidth of vertical_lines
    :param raster_face_color: the background color of the raster
    :param cell_spikes_color: the color of the spikes of the raster
    :param spike_shape: shape of the spike, "|", "*", "o"
    :param spike_shape_size: use for shape != of "|"
    :param seq_times_to_color_dict: None or a dict with as the key a tuple of int representing the cell index,
    and as a value a list of set, each set composed of int representing the times value at which the cell spike should
    be colored. It will be colored if there is indeed a spike at that time otherwise, the default color will be used.
    :param seq_colors: A dict, with key a tuple represening the indices of the seq and as value of colors,
    a color, should have the same keys as seq_times_to_color_dict
    :param link_seq_color: if not None, give the color with which link the spikes from a sequence. If not None,
    seq_colors will be ignored. could be a dict with key same tuple as seq_times_to_color_dict or a string and then
    we use the same color for all seq
    :param min_len_links_seq: minimum len of a seq for the links to be drawn
    :param axes_list if not None, give a list of axes that will be used, and be filled, but no figure will be created
    or saved then. Doesn't work yet is show_amplitude is True
    :param SCE_times:  a list of tuple corresponding to the first and last index of each SCE,
    (last index being included in the SCE). Will display the position of the SCE and their number above the activity
    diagram. If None, the overall time will be displayed. Need to be adapted to the format spike_numw or
    spike_train. Equivalent to span_are_coords
    :param without_activity_sum: if True, don't plot the sum of activity diagram, valid only if axes_list is not None
    :param spike_nums_for_activity_sum: if different that the one given for the raster, should be the
    same second dimension
    :param spikes_sum_to_use: an array of 1D, that will be use to display the sum of activity,
    :param size_fig: tuple of int
    :param cmap_name: "jet" by default, used if with_amplitude for the colormap
    :param traces if not None and display_traces is True, will display traces instead of a raster
    :param display_traces, if True display traces
    :param display_spike_nums, if False, won't display a raster using spike_nums
    :param traces_lw, default 0.3,  linewidth of the traces
    :param path_results: indicate where to save the plot, replace the param.path_results if it exists
    :param desaturate_color_according_to_normalized_amplitude: if True, spike_nums should be filled with float between
    0 and 1, representing the amplitude of the spike. And if a color is given for a cell, then it will be desaturate
    according to this value
    :param lines_to_display, dict that takes for a key a tuple of int representing 2 cells, and as value a list of tuple of 2 float
    representing the 2 extremities of a line between those 2 cells. By defualt, no lines
    :param lines_color="white": colors of lines_to_display
    :param lines_width=1: width of lines_to_display
    :param lines_band=0: if > 0, display a band around the line with transparency
    :param lines_band_color="white"
    :param xkcd_mode (bool): Turn on xkcd sketch-style drawing mode
    :param with_timestamp_in_file_name: add a time_stamps in the file saved to identify when it's done
    :return:
    """

    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    # + 11 diverting
    brewer_colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                     '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
                     '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
                     '#74add1', '#4575b4', '#313695']
    brewer_colors = brewer_colors[::-1]

    if (spike_nums is None) and (traces is None):
        return

    if display_traces:
        if traces is None:
            return

    if spike_nums_for_activity_sum is None:
        if spike_nums is None:
            without_activity_sum = True
        else:
            spike_nums_for_activity_sum = spike_nums

    if plot_with_amplitude and spike_train_format:
        # not possible for now
        return

    if xkcd_mode:
        plt.xkcd()

    if spike_nums is None:
        n_cells = len(traces)
    else:
        n_cells = len(spike_nums)

    if axes_list is None:
        if size_fig is None:
            size_fig = (15, 8)
        if not plot_with_amplitude:
            if without_activity_sum:
                fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=False,
                                        figsize=size_fig, dpi=dpi)
            else:
                sharex = True  # False if (SCE_times is None) else True
                fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=sharex,
                                               gridspec_kw={'height_ratios': [10, 2]},
                                               figsize=size_fig, dpi=dpi)
            fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 2.0, 'h_pad': 1.5})
        else:
            fig = plt.figure(figsize=size_fig, dpi=dpi)
            fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'h_pad': 1})
            outer = gridspec.GridSpec(1, 2, width_ratios=[100, 1])  # , wspace=0.2, hspace=0.2)
        fig.patch.set_facecolor(figure_background_color)
    else:
        if without_activity_sum:
            ax1 = axes_list[0]
        else:
            ax1, ax2 = axes_list

    if plot_with_amplitude:
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                 subplot_spec=outer[0], height_ratios=[10, 2])
        # inner.tight_layout(fig, pad=0.1)
        ax1 = fig.add_subplot(inner[0])  # plt.Subplot(fig, inner[0])
        min_value = np.min(spike_nums)
        max_value = np.max(spike_nums)
        step_color_value = 0.1
        colors = np.r_[np.arange(min_value, max_value, step_color_value)]
        mymap = plt.get_cmap("jet")
        # get the colors from the color map
        my_colors = mymap(colors)

        # colors = plt.cm.hsv(y / float(max(y)))
        scalar_map = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=min_value, vmax=max_value))
        # # fake up the array of the scalar mappable. Urgh…
        scalar_map._A = []

    # -------- end plot with amplitude ---------

    ax1.set_facecolor(raster_face_color)

    min_time = 0
    max_time = 0

    max_n_color = 10
    if display_traces:
        n_times = len(traces[0, :])
        zorder_traces = 21 + len(traces)
        for cell, trace in enumerate(traces):
            if use_brewer_colors_for_traces:
                color = brewer_colors[cell % len(brewer_colors)]
            else:
                color = cm.nipy_spectral(((cell % max_n_color) + 1) / (max_n_color + 1))
            ax1.plot(np.arange(n_times), trace + cell, lw=traces_lw, color=color, zorder=zorder_traces)
            if scatters_on_traces is not None:
                times_to_scatter = np.where(scatters_on_traces[cell, :])[0]
                ax1.scatter(times_to_scatter, trace[times_to_scatter] + cell,
                            color="white",
                            marker=scatters_on_traces_marker,
                            s=scatters_on_traces_size, zorder=zorder_traces - 1)
            zorder_traces -= 1
            line_beg_x = -1
            line_end_x = n_times + 1
            if display_dashed_line_with_traces:
                ax1.hlines(cell, line_beg_x, line_end_x, lw=0.1, linestyles="dashed", color=color, zorder=5)
    if cells_to_highlight is not None:
        cells_to_highlight = np.array(cells_to_highlight)

    if display_spike_nums:
        for cell, spikes in enumerate(spike_nums):
            if len(spikes) == 0:
                continue
            if spike_train_format:
                if cell == 0:
                    min_time = np.min(spikes)
                else:
                    min_time = int(np.min((min_time, np.min(spikes))))
                max_time = int(np.ceil(np.max((max_time, np.max(spikes)))))
            # print(f"Neuron {y}, total spikes {len(np.where(neuron)[0])}, "
            #       f"nb > 2: {len(np.where(neuron>2)[0])}, nb < 2: {len(np.where(neuron[neuron<2])[0])}")
            if display_traces:
                # same color as traces
                color_neuron = cm.nipy_spectral(((cell % max_n_color) + 1) / (max_n_color + 1))
            else:
                color_neuron = cell_spikes_color
            if cells_to_highlight is not None:
                if cell in cells_to_highlight:
                    index = np.where(cells_to_highlight == cell)[0][0]
                    color_neuron = cells_to_highlight_colors[index]
            if spike_train_format:
                neuron_times = spikes
            else:
                neuron_times = np.where(spikes > 0)[0]
            if spike_shape != "|":
                if plot_with_amplitude:
                    ax1.scatter(neuron_times, np.repeat(cell, len(neuron_times)),
                                color=scalar_map.to_rgba(spikes[spikes > 0]),
                                marker=spike_shape,
                                s=spike_shape_size, zorder=20)
                elif desaturate_color_according_to_normalized_amplitude:
                    n_spikes = len(neuron_times)
                    colors_list = [sns.desaturate(x, p) for x, p in
                                   zip([color_neuron] * n_spikes, spikes[neuron_times])]
                    ax1.scatter(neuron_times, np.repeat(cell, len(neuron_times)),
                                color=colors_list,
                                marker=spike_shape,
                                s=spike_shape_size, zorder=20)
                else:
                    if display_traces:
                        y_values = traces[cell, neuron_times] + cell
                    else:
                        y_values = np.repeat(cell, len(neuron_times))
                    ax1.scatter(neuron_times, y_values, color=color_neuron, marker=spike_shape,
                                s=spike_shape_size, zorder=20)
            else:
                if desaturate_color_according_to_normalized_amplitude:
                    n_spikes = len(neuron_times)
                    colors_list = [sns.desaturate(x, p) for x, p in
                                   zip([color_neuron] * n_spikes, spikes[neuron_times])]
                    ax1.vlines(neuron_times, cell - .5, cell + .5, color=colors_list,
                               linewidth=spike_shape_size, zorder=20)
                elif plot_with_amplitude:
                    ax1.vlines(neuron_times, cell - .5, cell + .5, color=scalar_map.to_rgba(spikes[spikes > 0]),
                               linewidth=spike_shape_size, zorder=20)
                else:
                    ax1.vlines(neuron_times, cell - .5, cell + .5, color=color_neuron, linewidth=spike_shape_size,
                               zorder=20)
    if lines_to_display is not None:
        """
        lines_to_display=None,
                       lines_color="white",
                       lines_width=1,
                       lines_band=0,
                       lines_band_color="white"
                       dict that takes for a key a tuple of int representing 2 cells, and as value a list of tuple of 2 float
    representing the 2 extremities of a line between those 2 cells. By defualt, no lines
        """
        for cells_tuple, spike_times_list in lines_to_display.items():
            for spike_times in spike_times_list:
                ax1.plot(list(spike_times), list(cells_tuple),
                         color=lines_color,
                         linewidth=lines_width, zorder=30, alpha=1)
                if lines_band > 0:
                    xy = np.zeros((4, 2))
                    xy[0, 0] = spike_times[0] - lines_band
                    xy[0, 1] = cells_tuple[0]
                    xy[1, 0] = spike_times[1] - lines_band
                    xy[1, 1] = cells_tuple[1]
                    xy[2, 0] = spike_times[1] + lines_band
                    xy[2, 1] = cells_tuple[1]
                    xy[3, 0] = spike_times[0] + lines_band
                    xy[3, 1] = cells_tuple[0]
                    band_patch = patches.Polygon(xy=xy,
                                                 fill=True,
                                                 # linewidth=line_width,
                                                 facecolor=lines_band_color,
                                                 # edgecolor=edge_color,
                                                 alpha=0.4,
                                                 zorder=10)  # lw=2
                    ax1.add_patch(band_patch)
                    # ax1.fill_between(list(spike_times), np.array(cells_tuple)-lines_band,
                    #                  np.array(cells_tuple)+lines_band, facecolor=lines_band_color, alpha=0.4,
                    #                  zorder=10)
    if seq_times_to_color_dict is not None:
        seq_count = 0
        links_labels = []
        links_labels_color = []
        links_labels_y_coord = []
        if jitter_links_range > 0:
            nb_jitters = 10
            indices_rand_x = np.linspace(-jitter_links_range, jitter_links_range, nb_jitters)
            np.random.shuffle(indices_rand_x)
        for seq_indices, seq_times_list in seq_times_to_color_dict.items():
            nb_seq_times = 0
            for times_list_index, times_list in enumerate(seq_times_list):
                x_coord_to_link = []
                y_coord_to_link = []
                for time_index, t in enumerate(times_list):
                    cell_index = seq_indices[time_index]
                    # in case of this seq of cell would be used in a zoom version of the raster
                    if cell_index >= n_cells:
                        continue
                    # first we make sure the cell does spike at the given time
                    if spike_train_format:
                        if t not in spike_nums[cell_index]:
                            continue
                    else:
                        pass
                        if spike_nums[cell_index, t] == 0:
                            cell_for_msg = cell_index
                            if y_ticks_labels is not None:
                                cell_for_msg = y_ticks_labels[cell_index]
                            print(f"Not there: seq {times_list_index} cell {cell_for_msg} - {cell_index}, "
                                  f"time {t}")
                            continue
                        # print(f"## There: seq {times_list_index} cell {cell_index}, time {t}")
                    if link_seq_color is not None:
                        x_coord_to_link.append(t)
                        y_coord_to_link.append(cell_index)
                    else:
                        # if so, we draw the spike
                        if spike_shape != "|":
                            ax1.scatter(t, cell_index, color=seq_colors[seq_indices],
                                        marker=spike_shape,
                                        s=spike_shape_size, zorder=20)
                        else:
                            ax1.vlines(t, cell_index - .5, cell_index + .5, color=seq_colors[seq_indices],
                                       linewidth=1, zorder=20)
                if (link_seq_color is not None) and (len(x_coord_to_link) >= min_len_links_seq):
                    if isinstance(link_seq_color, str):
                        color_to_use = link_seq_color
                    elif isinstance(link_seq_color, dict):
                        color_to_use = link_seq_color[seq_indices]
                    else:
                        color_to_use = link_seq_color[seq_count % len(link_seq_color)]
                    x_coord_to_link = np.array(x_coord_to_link)
                    if jitter_links_range > 0:
                        jitter_to_add = indices_rand_x[seq_count % nb_jitters]
                    else:
                        jitter_to_add = 0
                    ax1.plot(x_coord_to_link + jitter_to_add, y_coord_to_link,
                             color=color_to_use,
                             linewidth=link_seq_line_width, zorder=30, alpha=link_seq_alpha)
                    nb_seq_times += 1
            if nb_seq_times > 0:
                category = ""
                if link_seq_categories is not None:
                    category = "*" * link_seq_categories[seq_indices]
                links_labels.append(f"l{len(seq_indices)}, r{nb_seq_times} {category}")
                links_labels_color.append(color_to_use)
                links_labels_y_coord.append((seq_indices[0] + seq_indices[-1]) / 2)
            seq_count += 1

    if display_traces:
        ax1.set_ylim(-0.5, n_cells + 4)
    else:
        ax1.set_ylim(-0.5, n_cells)
    if y_ticks_labels is not None:
        ax1.set_yticks(np.arange(n_cells))
        ax1.set_yticklabels(y_ticks_labels)
    if (x_ticks_labels is not None) and (x_ticks is not None):
        ax1.set_xticks(x_ticks)
        ax1.tick_params('x', length=2, width=0.5, which='both')
        ax1.set_xticklabels(x_ticks_labels, rotation=45)  # ha="right", va="center
    if x_ticks_labels_size is not None:
        ax1.xaxis.set_tick_params(labelsize=x_ticks_labels_size)
    if y_ticks_labels_size is not None:
        ax1.yaxis.set_tick_params(labelsize=y_ticks_labels_size)
    else:
        if n_cells < 50:
            y_ticks_labels_size = 5
        elif n_cells < 100:
            y_ticks_labels_size = 4
        elif n_cells < 200:
            y_ticks_labels_size = 3
        elif n_cells < 400:
            y_ticks_labels_size = 1
        else:
            y_ticks_labels_size = 0.1
        ax1.yaxis.set_tick_params(labelsize=y_ticks_labels_size)

    if hide_raster_y_ticks_labels:
        ax1.tick_params(labelleft=False)

    if without_ticks:
        if x_ticks is not None:
            ax1.tick_params(axis='y', which='both', length=0)
        else:
            ax1.tick_params(axis='both', which='both', length=0)

    if seq_times_to_color_dict is not None:
        if link_seq_color is not None:
            ax_right = ax1.twinx()
            ax_right.set_frame_on(False)
            ax_right.set_ylim(-1, n_cells)
            ax_right.set_yticks(links_labels_y_coord)
            # clusters labels
            ax_right.set_yticklabels(links_labels)
            ax_right.yaxis.set_ticks_position('none')
            if y_ticks_labels_size > 1:
                y_ticks_labels_size -= 1
            else:
                y_ticks_labels_size -= 0.5
            ax_right.yaxis.set_tick_params(labelsize=y_ticks_labels_size)
            # ax_right.yaxis.set_tick_params(labelsize=2)
            for index in np.arange(len(links_labels)):
                ax_right.get_yticklabels()[index].set_color(links_labels_color[index])

    if spike_train_format:
        n_times = int(math.ceil(max_time - min_time))
    else:
        if spike_nums is None:
            n_times = traces.shape[1]
        else:
            n_times = len(spike_nums[0, :])

    # draw span to highlight some periods
    if span_area_coords is not None and (len(span_area_coords) > 0) and (span_area_colors is not None) and \
            (len(span_area_coords) == len(span_area_colors)):
        # if len(span_area_coords) != len(span_area_colors):
        #     raise Exception("span_area_coords and span_area_colors are not the same size")
        for index, span_area_coord in enumerate(span_area_coords):
            for coord in span_area_coord:
                if span_area_colors is not None:
                    color = span_area_colors[index]
                else:
                    color = "lightgrey"
                ax1.axvspan(coord[0], coord[1], alpha=alpha_span_area, facecolor=color, zorder=1)

    if (span_cells_to_highlight is not None):
        for index, cell_to_span in enumerate(span_cells_to_highlight):
            ax1.axhspan(cell_to_span - 0.5, cell_to_span + 0.5, alpha=0.4,
                        facecolor=span_cells_to_highlight_colors[index])

    if horizontal_lines is not None:
        line_beg_x = 0
        line_end_x = 0
        if spike_train_format:
            line_beg_x = min_time - 1
            line_end_x = max_time + 1
        else:
            line_beg_x = -1
            line_end_x = n_times + 1
        if horizontal_lines_linewidth is None:
            ax1.hlines(horizontal_lines, line_beg_x, line_end_x, color=horizontal_lines_colors, linewidth=2,
                       linestyles=horizontal_lines_sytle)
        else:
            ax1.hlines(horizontal_lines, line_beg_x, line_end_x, color=horizontal_lines_colors,
                       linewidth=horizontal_lines_linewidth,
                       linestyles=horizontal_lines_sytle)

    if vertical_lines is not None:
        line_beg_y = 0
        line_end_y = n_cells - 1
        ax1.vlines(vertical_lines, line_beg_y, line_end_y, color=vertical_lines_colors,
                   linewidth=vertical_lines_linewidth,
                   linestyles=vertical_lines_sytle)

    if spike_train_format:
        ax1.set_xlim(min_time - 1, max_time + 1)
    else:
        ax1.set_xlim(-1, n_times + 1)
    # ax1.margins(x=0, tight=True)

    if not without_activity_sum or hide_x_ticks_labels:
        ax1.get_xaxis().set_visible(False)

    if title is not None:
        ax1.set_title(title)
    # Give x axis label for the spike raster plot
    # ax.xlabel('Frames')
    # Give y axis label for the spike raster plot
    if raster_y_axis_label is not None:
        ax1.set_ylabel(raster_y_axis_label)
    if raster_y_axis_label_size is not None:
        ax1.yaxis.label.set_size(raster_y_axis_label_size)
    ax1.xaxis.label.set_color(axes_label_color)
    ax1.yaxis.label.set_color(axes_label_color)

    # ax1.spines['left'].set_color(y_ticks_labels_color)
    # ax1.spines['bottom'].set_color(x_ticks_labels_color)
    # ax1.yaxis.label.set_color(y_ticks_labels_color)
    if isinstance(y_ticks_labels_color, list):
        for xtick, color in zip(ax1.get_yticklabels(), y_ticks_labels_color):
            xtick.set_color(color)
    else:
        ax1.tick_params(axis='y', colors=y_ticks_labels_color)
    ax1.tick_params(axis='x', colors=x_ticks_labels_color)

    # removing frame
    ax1.set_frame_on(False)

    if traces is not None and display_traces:
        ax1.set_ylim(-3, len(traces) + 10)

    if (axes_list is not None) and without_activity_sum:
        return

    if not without_activity_sum:
        # ################################################################################################
        # ################################ Activity sum plot part ################################
        # ################################################################################################
        if (sliding_window_duration >= 1) and (spikes_sum_to_use is None):
            sum_spikes = np.zeros(n_times)
            if spike_train_format:
                windows_sum = np.zeros((n_cells, n_times), dtype="int16")
                # one cell can participate to max one spike by window
                # if value is True, it means this cell has already been counted
                cell_window_participation = np.zeros((n_cells, n_times), dtype="bool")
                for cell, spikes_train in enumerate(spike_nums_for_activity_sum):
                    for spike_time in spikes_train:
                        # first determining to which windows to add the spike
                        spike_index = int(spike_time - min_time)
                        first_index_window = np.max((0, spike_index - sliding_window_duration))
                        if np.sum(cell_window_participation[cell, first_index_window:spike_index]) == 0:
                            windows_sum[cell, first_index_window:spike_index] += 1
                            cell_window_participation[cell, first_index_window:spike_index] = True
                        else:
                            for t in np.arange(first_index_window, spike_index):
                                if cell_window_participation[cell, t] is False:
                                    windows_sum[cell, t] += 1
                                    cell_window_participation[cell, t] = True
                sum_spikes = np.sum(windows_sum, axis=0)
                if debug_mode:
                    print("sliding window over")
                # for index, t in enumerate(np.arange(int(min_time), int((np.ceil(max_time) - sliding_window_duration)))):
                #     # counting how many cell fire during that window
                #     if (index % 1000) == 0:
                #         print(f"index {index}")
                #     sum_value = 0
                #     t_min = t
                #     t_max = t + sliding_window_duration
                #     for spikes_train in spike_nums:
                #         # give the indexes
                #         # np.where(np.logical_and(spikes_train >= t, spikes_train < t_max))
                #         spikes = spikes_train[np.logical_and(spikes_train >= t, spikes_train < t_max)]
                #         nb_spikes = len(spikes)
                #         if nb_spikes > 0:
                #             sum_value += 1
                #     sum_spikes[index] = sum_value
                # sum_spikes[(n_times - sliding_window_duration):] = sum_value
            else:
                sum_value = None
                for t in np.arange(0, (n_times - sliding_window_duration)):
                    # One spike by cell max in the sum process
                    sum_value = np.sum(spike_nums_for_activity_sum[:, t:(t + sliding_window_duration)], axis=1)
                    sum_spikes[t] = len(np.where(sum_value)[0])
                if sum_value is not None:
                    sum_spikes[(n_times - sliding_window_duration):] = len(np.where(sum_value)[0])
        elif spikes_sum_to_use is None:
            if spike_train_format:
                pass
            else:
                # binary_spikes = np.zeros((n_cells, n_times), dtype="int8")
                # for spikes, spikes in enumerate(spike_nums_for_activity_sum):
                #     binary_spikes[spikes, spikes > 0] = 1
                # # if (param is not None) and (param.bin_size > 1):
                # #     sum_spikes = np.mean(np.split(np.sum(binary_spikes, axis=0), n_times // param.bin_size), axis=1)
                # #     sum_spikes = np.repeat(sum_spikes, param.bin_size)
                # # else:
                sum_spikes = np.sum(spike_nums_for_activity_sum, axis=0)
        else:
            sum_spikes = spikes_sum_to_use

        if spike_train_format:
            x_value = np.arange(min_time, max_time)
        else:
            x_value = np.arange(n_times)

        if plot_with_amplitude:
            ax2 = fig.add_subplot(inner[1], sharex=ax1)

        ax2.set_facecolor(activity_sum_face_color)

        # sp = UnivariateSpline(x_value, sum_spikes, s=240)
        # ax2.fill_between(x_value, 0, smooth_curve(sum_spikes), facecolor="black") # smooth_curve(sum_spikes)
        if show_sum_spikes_as_percentage:
            if debug_mode:
                print("using percentages")
            sum_spikes = sum_spikes / n_cells
            sum_spikes *= 100
            if activity_threshold is not None:
                activity_threshold = activity_threshold / n_cells
                activity_threshold *= 100

        ax2.fill_between(x_value, 0, sum_spikes, facecolor=activity_sum_plot_color, zorder=10)
        if activity_threshold is not None:
            line_beg_x = 0
            line_end_x = 0
            if spike_train_format:
                line_beg_x = min_time - 1
                line_end_x = max_time + 1
            else:
                line_beg_x = -1
                line_end_x = len(spike_nums_for_activity_sum[0, :]) + 1
            ax2.hlines(activity_threshold, line_beg_x, line_end_x, color="red", linewidth=1, linestyles="dashed")

        # draw span to highlight some periods
        if (span_area_coords is not None) and (not span_area_only_on_raster):
            for index, span_area_coord in enumerate(span_area_coords):
                for coord in span_area_coord:
                    if span_area_colors is not None:
                        color = span_area_colors[index]
                    else:
                        color = "lightgrey"
                    ax2.axvspan(coord[0], coord[1], alpha=0.5, facecolor=color, zorder=1)

        # early born
        if (cells_to_highlight is not None) and color_peaks_activity:
            for index, cell_to_span in enumerate(cells_to_highlight):
                ax2.vlines(np.where(spike_nums_for_activity_sum[cell_to_span, :])[0], 0, np.max(sum_spikes),
                           color=cells_to_highlight_colors[index],
                           linewidth=2, linestyles="dashed", alpha=0.2)

        # ax2.yaxis.set_visible(False)
        ax2.set_frame_on(False)
        if hide_x_ticks_labels:
            ax2.get_xaxis().set_visible(False)
        else:
            ax2.get_xaxis().set_visible(True)

        if y_lim_sum_activity is not None:
            ax2.set_ylim(y_lim_sum_activity[0], y_lim_sum_activity[1])
        else:
            if len(sum_spikes) > 0:
                ax2.set_ylim(0, np.max(sum_spikes))
        if spike_train_format:
            ax2.set_xlim(min_time - 1, max_time + 1)
        else:
            if spike_nums_for_activity_sum is not None:
                ax2.set_xlim(-1, len(spike_nums_for_activity_sum[0, :]) + 1)
            else:
                ax2.set_xlim(-1, len(spikes_sum_to_use) + 1)

        if SCE_times is not None:
            ax_top = ax2.twiny()
            ax_top.set_frame_on(False)
            if spike_train_format:
                ax_top.set_xlim(min_time - 1, max_time + 1)
            else:
                if spike_nums_for_activity_sum is not None:
                    ax_top.set_xlim(-1, len(spike_nums_for_activity_sum[0, :]) + 1)
                else:
                    ax_top.set_xlim(-1, len(spikes_sum_to_use) + 1)
            xticks_pos = []
            for times_tuple in SCE_times:
                xticks_pos.append(times_tuple[0])
            ax_top.set_xticks(xticks_pos)
            ax_top.xaxis.set_ticks_position('none')
            ax_top.set_xticklabels(np.arange(len(SCE_times)))
            ax_top.tick_params(axis='x', colors=x_ticks_labels_color)
            plt.setp(ax_top.xaxis.get_majorticklabels(), rotation=90)
            if len(SCE_times) > 30:
                ax_top.xaxis.set_tick_params(labelsize=3)
            elif len(SCE_times) > 50:
                ax_top.xaxis.set_tick_params(labelsize=2)
            elif len(SCE_times) > 100:
                ax_top.xaxis.set_tick_params(labelsize=1)
            elif len(SCE_times) > 300:
                ax_top.xaxis.set_tick_params(labelsize=0.5)
            else:
                ax_top.xaxis.set_tick_params(labelsize=4)
        # print(f"max sum_spikes {np.max(sum_spikes)}, mean  {np.mean(sum_spikes)}, median {np.median(sum_spikes)}")

        if without_ticks:
            ax2.tick_params(axis='both', which='both', length=0)
        # ax2.yaxis.label.set_color(y_ticks_labels_color)
        if isinstance(y_ticks_labels_color, list):
            for xtick, color in zip(ax2.get_yticklabels(), y_ticks_labels_color):
                xtick.set_color(color)
        else:
            ax2.tick_params(axis='y', colors=y_ticks_labels_color)
        ax2.tick_params(axis='x', colors=x_ticks_labels_color)

        if (x_ticks_labels is not None) and (x_ticks is not None):
            ax2.set_xticks(x_ticks)
            ax2.tick_params('x', length=2, width=0.5, which='both')
            ax2.set_xticklabels(x_ticks_labels, rotation=45)  # ha="right", va="center
        if x_ticks_labels_size is not None:
            ax2.xaxis.set_tick_params(labelsize=x_ticks_labels_size)
        ax2.xaxis.label.set_color(axes_label_color)
        ax2.yaxis.label.set_color(axes_label_color)

    # color bar section
    if plot_with_amplitude:
        inner_2 = gridspec.GridSpecFromSubplotSpec(1, 1,
                                                   subplot_spec=outer[1])  # , wspace=0.1, hspace=0.1)
        ax3 = fig.add_subplot(inner_2[0])  # plt.Subplot(fig, inner_2[0])
        cb = fig.colorbar(scalar_map, cax=ax3)
        cb.ax.tick_params(axis='y', colors="white")
    if axes_list is None:
        if save_raster and (path_results is not None):
            # transforming a string in a list
            if isinstance(save_formats, str):
                save_formats = [save_formats]
            time_str = ""
            if with_timestamp_in_file_name:
                time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
            for save_format in save_formats:
                if not with_timestamp_in_file_name:
                    fig.savefig(os.path.join(f'{path_results}', f'{file_name}.{save_format}'),
                                format=f"{save_format}",
                                facecolor=fig.get_facecolor())
                else:
                    fig.savefig(os.path.join(f'{path_results}', f'{file_name}_{time_str}.{save_format}'),
                                format=f"{save_format}",
                                facecolor=fig.get_facecolor())
        # Display the spike raster plot
        if show_raster:
            plt.show()
        plt.close()
