import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def plot_scatter_family(data_dict, colors_dict,
                        filename,
                        y_label,
                        path_results=None, y_lim=None,
                        x_label=None,
                        y_log=False,
                        scatter_size=200,
                        scatter_alpha=1,
                        background_color="black",
                        link_scatter=False,
                        labels_color="white",
                        with_x_jitter=0.2,
                        with_y_jitter=None,
                        x_labels_rotation=None,
                        save_formats="pdf",
                        with_timestamp_in_file_name=True):
    """

    :param data_dict: key is a label, value is a list of 3 list of int, of same size, first one are the x-value,
    second one is the y-values and third one is the number of elements that allows to get this number (like number
    of sessions)
    :param colors_dict = key is a label, value is a color
    :param filename:
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

    for label, data_to_scatters in data_dict.items():
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

        ax1.scatter(x_pos, y_pos,
                    color=colors_dict[label],
                    alpha=scatter_alpha,
                    marker="o",
                    edgecolors=labels_color,
                    label=label,
                    s=scatter_size, zorder=21)

        if link_scatter:
            ax1.plot(x_pos, y_pos, zorder=30, color=colors_dict[label],
                     linewidth=2)

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
