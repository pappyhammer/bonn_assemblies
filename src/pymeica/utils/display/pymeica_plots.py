import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def plot_scatter_family(data_dict, colors_dict,
                        filename,
                        y_label,
                        label_to_legend=None,
                        path_results=None, y_lim=None,
                        x_label=None,
                        y_log=False,
                        scatter_size=200,
                        scatter_alpha=1,
                        background_color="black",
                        lines_plot_values=None,
                        link_scatter=False,
                        labels_color="white",
                        with_x_jitter=0.2,
                        with_y_jitter=None,
                        x_labels_rotation=None,
                        h_lines_y_values=None,
                        save_formats="pdf",
                        with_timestamp_in_file_name=True):
    """
    Plot family of scatters (same color and label) with possibly lines that are associated to it.
    :param data_dict: key is a label, value is a list of 3 list of int, of same size, first one are the x-value,
    second one is the y-values and third one is the number of elements that allows to get this number (like number
    of sessions)
    :param colors_dict = key is a label, value is a color
    :param label_to_legend: (dict) if not None, key are label of data_dict, value is the label to be displayed as legend
    :param filename:
    :param lines_plot_values: (dict) same keys as data_dict, value is a list of list of 2 list of int or float,
    representing x & y value of plot to trace
    :param y_label:
    :param y_lim: tuple of int,
    :param link_scatter: draw a line between scatters
    :param save_formats:
    :return:
    """
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(12, 12))

    ax1.set_facecolor(background_color)

    fig.patch.set_facecolor(background_color)

    # labels = []
    # data_list = []
    # scatter_text_list = []
    # medians_values = []
    min_x_value = 10000
    max_x_value = 0

    for label, data_to_scatters in data_dict.items():
        min_x_value = min(min_x_value, np.min(data_to_scatters[0]))
        max_x_value = max(max_x_value, np.max(data_to_scatters[0]))

        # Adding jitter
        if (with_x_jitter > 0) and (with_x_jitter < 1):
            x_pos = [x + ((np.random.random_sample() - 0.5) * with_x_jitter) for x in data_to_scatters[0]]
        else:
            x_pos = data_to_scatters[0]

        if with_y_jitter is not None:
            y_pos = [y + (((np.random.random_sample() - 0.5) * 2) * with_y_jitter) for y in data_to_scatters[1]]
        else:
            y_pos = data_to_scatters[1]

        colors_scatters = []
        while len(colors_scatters) < len(y_pos):
            colors_scatters.extend(colors_dict[label])

        label_legend = label_to_legend[label] if label_to_legend is not None else label
        ax1.scatter(x_pos, y_pos,
                    color=colors_dict[label],
                    alpha=scatter_alpha,
                    marker="o",
                    edgecolors=labels_color,
                    label=label_legend,
                    s=scatter_size, zorder=21)

        if link_scatter:
            ax1.plot(x_pos, y_pos, zorder=30, color=colors_dict[label],
                     linewidth=1)

        if lines_plot_values is not None:
            if label in lines_plot_values:
                for lines_coordinates in lines_plot_values[label]:
                    x_pos, y_pos = lines_coordinates
                    ax1.plot(x_pos, y_pos, zorder=35, color=colors_dict[label],
                             linewidth=2)

    if h_lines_y_values is not None:
        for y_value in h_lines_y_values:
            ax1.hlines(y_value, min_x_value,
                       max_x_value, color=labels_color, linewidth=0.5,
                       linestyles="dashed", zorder=25)

    ax1.set_ylabel(f"{y_label}", fontsize=30, labelpad=20)
    if y_lim is not None:
        ax1.set_ylim(y_lim[0], y_lim[1])
    if x_label is not None:
        ax1.set_xlabel(x_label, fontsize=30, labelpad=20)
    ax1.xaxis.label.set_color(labels_color)
    ax1.yaxis.label.set_color(labels_color)
    if y_log:
        ax1.set_yscale("log")

    ax1.legend()

    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.xaxis.set_tick_params(labelsize=15)
    ax1.tick_params(axis='y', colors=labels_color)
    ax1.tick_params(axis='x', colors=labels_color)
    # xticks = np.arange(1, len(data_dict) + 1)
    # ax1.set_xticks(xticks)
    # removing the ticks but not the labels
    # ax1.xaxis.set_ticks_position('none')
    # sce clusters labels
    # ax1.set_xticklabels(labels)
    # if x_labels_rotation is not None:
    #     for tick in ax1.get_xticklabels():
    #         tick.set_rotation(x_labels_rotation)

    # padding between ticks label and  label axis
    # ax1.tick_params(axis='both', which='major', pad=15)
    fig.tight_layout()
    # adjust the space between axis and the edge of the figure
    # https://matplotlib.org/faq/howto_faq.html#move-the-edge-of-an-axes-to-make-room-for-tick-labels
    # fig.subplots_adjust(left=0.2)

    if with_timestamp_in_file_name:
        time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")

    if isinstance(save_formats, str):
        save_formats = [save_formats]

    for save_format in save_formats:
        fig.savefig(f'{path_results}/{filename}'
                    f'_{time_str}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())

    plt.close()
