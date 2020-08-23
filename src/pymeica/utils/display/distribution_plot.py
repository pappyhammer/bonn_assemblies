import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import math
import os
import seaborn as sns
import pandas as pd


def plot_box_plots(data_dict, title, filename,
                   y_label,
                   scatter_text_dict=None,
                   colors=None,
                   path_results=None, y_lim=None,
                   x_label=None, with_scatters=True,
                   y_log=False,
                   scatters_with_same_colors=None,
                   scatter_size=20,
                   scatter_alpha=0.5,
                   n_sessions_dict=None,
                   background_color="black",
                   link_medians=False,
                   color_link_medians="red",
                   labels_color="white",
                   with_x_jitter=True,
                   with_y_jitter=None,
                   x_labels_rotation=None,
                   fliers_symbol=None,
                   save_formats="pdf",
                   with_timestamp_in_file_name=True):
    """

    :param data_dict:
    :param scatter_text_dict: same dimension of data_dict, for each value associate a string that will be displayed
    in the scatter if with_scatters is True
    :param colors: list of colors, if not None, the colors will be used to color the boxplot. If there are
    less colors than boxplot, then it will be looped
    :param n_sessions_dict: should be the same keys as data_dict, value is an int reprenseing the number of sessions
    that gave those data (N), a n will be display representing the number of poins in the boxplots if n != N
    :param title:
    :param filename:
    :param y_label:
    :param y_lim: tuple of int,
    :param scatters_with_same_colors: scatter that have the same index in the data_dict,, will be colors
    with the same colors, using the list of colors given by scatters_with_same_colors
    :param save_formats:
    :return:
    """
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(12, 12))

    colorfull = (colors is not None)

    median_color = background_color if colorfull else labels_color

    scatter_edgecolors = labels_color

    ax1.set_facecolor(background_color)

    fig.patch.set_facecolor(background_color)

    labels = []
    data_list = []
    scatter_text_list = []
    medians_values = []
    for age, data in data_dict.items():
        data_list.append(data)
        if scatter_text_dict is not None:
            scatter_text_list.append(scatter_text_dict[age])
        medians_values.append(np.median(data))
        label = age
        if n_sessions_dict is None:
            # label += f"\n(n={len(data)})"
            pass
        else:
            n_sessions = n_sessions_dict[age]
            if n_sessions != len(data):
                label += f"\n(N={n_sessions}, n={len(data)})"
            else:
                label += f"\n(N={n_sessions})"
        labels.append(label)
    sym = ""
    if fliers_symbol is not None:
        sym = fliers_symbol
    bplot = plt.boxplot(data_list, patch_artist=colorfull,
                        labels=labels, sym=sym, zorder=20)  # whis=[5, 95], sym='+'
    # color=["b", "cornflowerblue"],
    # fill with colors

    # edge_color="silver"

    for element in ['boxes', 'whiskers', 'fliers', 'caps']:
        plt.setp(bplot[element], color="white")

    for element in ['means', 'medians']:
        plt.setp(bplot[element], color=median_color)

    if colorfull:
        while len(colors) < len(data_dict):
            colors.extend(colors)
        colors = colors[:len(data_dict)]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            r, g, b, a = patch.get_facecolor()
            # for transparency purpose
            patch.set_facecolor((r, g, b, 0.8))

    if with_scatters:
        for data_index, data in enumerate(data_list):
            # Adding jitter
            if with_x_jitter:
                x_pos = [1 + data_index + ((np.random.random_sample() - 0.5) * 0.5) for x in np.arange(len(data))]
            else:
                x_pos = [1 + data_index for x in np.arange(len(data))]

            if with_y_jitter is not None:
                y_pos = [value + (((np.random.random_sample() - 0.5) * 2) * with_y_jitter) for value in data]
            else:
                y_pos = data

            colors_scatters = []
            if scatters_with_same_colors is not None:
                while len(colors_scatters) < len(y_pos):
                    colors_scatters.extend(scatters_with_same_colors)
            else:
                colors_scatters = [colors[data_index]]

            ax1.scatter(x_pos, y_pos,
                        color=colors_scatters[:len(y_pos)],
                        alpha=scatter_alpha,
                        marker="o",
                        edgecolors=scatter_edgecolors,
                        s=scatter_size, zorder=21)
            # plotting text in the scatter if given so
            if scatter_text_dict is not None:
                scatter_text_data = scatter_text_list[data_index]
                for scatter_index in np.arange(len(data)):
                    scatter_text = str(scatter_text_data[scatter_index])
                    if len(scatter_text) > 3:
                        font_size = 3
                    else:
                        font_size = 5
                    ax1.text(x=x_pos[scatter_index], y=y_pos[scatter_index],
                             s=scatter_text, color="black", zorder=22,
                             ha='center', va="center", fontsize=font_size, fontweight='bold')
    if link_medians:
        ax1.plot(np.arange(1, len(medians_values) + 1), medians_values, zorder=30, color=color_link_medians,
                 linewidth=2)

    # plt.xlim(0, 100)
    plt.title(title)

    ax1.set_ylabel(f"{y_label}", fontsize=30, labelpad=20)
    if y_lim is not None:
        ax1.set_ylim(y_lim[0], y_lim[1])
    if x_label is not None:
        ax1.set_xlabel(x_label, fontsize=30, labelpad=20)
    ax1.xaxis.label.set_color(labels_color)
    ax1.yaxis.label.set_color(labels_color)
    if y_log:
        ax1.set_yscale("log")

    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.xaxis.set_tick_params(labelsize=15)
    ax1.tick_params(axis='y', colors=labels_color)
    ax1.tick_params(axis='x', colors=labels_color)
    xticks = np.arange(1, len(data_dict) + 1)
    ax1.set_xticks(xticks)
    # removing the ticks but not the labels
    ax1.xaxis.set_ticks_position('none')
    # sce clusters labels
    ax1.set_xticklabels(labels)
    if x_labels_rotation is not None:
        for tick in ax1.get_xticklabels():
            tick.set_rotation(x_labels_rotation)

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


def plot_hist_distribution(distribution_data,
                           filename=None,
                           values_to_scatter=None,
                           n_bins=None,
                           use_log=False,
                           x_range=None,
                           labels=None,
                           scatter_shapes=None,
                           colors=None,
                           tight_x_range=False,
                           twice_more_bins=False,
                           scale_them_all=False,
                           background_color="black",
                           hist_facecolor="white",
                           hist_edgeccolor="white",
                           axis_labels_color="white",
                           axis_color="white",
                           axis_label_font_size=20,
                           ticks_labels_color="white",
                           ticks_label_size=14,
                           xlabel=None,
                           ylabel=None,
                           fontweight=None,
                           fontfamily=None,
                           size_fig=None,
                           dpi=100,
                           path_results=None,
                           save_formats="pdf",
                           ax_to_use=None,
                           color_to_use=None, legend_str=None,
                           density=False,
                           save_figure=False,
                           with_timestamp_in_file_name=True,
                           max_value=None):
    """
    Plot a distribution in the form of an histogram, with option for adding some scatter values
    :param distribution_data:
    :param description:
    :param param:
    :param values_to_scatter:
    :param labels:
    :param scatter_shapes:
    :param colors:
    :param tight_x_range:
    :param twice_more_bins:
    :param xlabel:
    :param ylabel:
    :param save_formats:
    :return:
    """
    distribution = np.array(distribution_data)

    if x_range is not None:
        min_range = x_range[0]
        max_range = x_range[1]
    elif tight_x_range:
        max_range = np.max(distribution)
        min_range = np.min(distribution)
    else:
        max_range = 100
        min_range = 0
    weights = (np.ones_like(distribution) / (len(distribution))) * 100
    # weights=None

    if ax_to_use is None:
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                gridspec_kw={'height_ratios': [1]},
                                figsize=size_fig, dpi=dpi)
        ax1.set_facecolor(background_color)
        fig.patch.set_facecolor(background_color)
    else:
        ax1 = ax_to_use
    if n_bins is not None:
        bins = n_bins
    else:
        bins = int(np.sqrt(len(distribution)))
        if twice_more_bins:
            bins *= 2

    hist_color = hist_facecolor
    if bins > 100:
        edge_color = hist_color
    else:
        edge_color = hist_edgeccolor
    ax1.spines['bottom'].set_color(axis_color)
    ax1.spines['left'].set_color(axis_color)

    hist_plt, edges_plt, patches_plt = ax1.hist(distribution, bins=bins, range=(min_range, max_range),
                                                facecolor=hist_color, log=use_log,
                                                edgecolor=edge_color, label=legend_str,
                                                weights=weights, density=density)
    if values_to_scatter is not None:
        scatter_bins = np.ones(len(values_to_scatter), dtype="int16")
        scatter_bins *= -1

        for i, edge in enumerate(edges_plt):
            # print(f"i {i}, edge {edge}")
            if i >= len(hist_plt):
                # means that scatter left are on the edge of the last bin
                scatter_bins[scatter_bins == -1] = i - 1
                break

            if len(values_to_scatter[values_to_scatter <= edge]) > 0:
                if (i + 1) < len(edges_plt):
                    bool_list = values_to_scatter < edge  # edges_plt[i + 1]
                    for i_bool, bool_value in enumerate(bool_list):
                        if bool_value:
                            if scatter_bins[i_bool] == -1:
                                new_i = max(0, i - 1)
                                scatter_bins[i_bool] = new_i
                else:
                    bool_list = values_to_scatter < edge
                    for i_bool, bool_value in enumerate(bool_list):
                        if bool_value:
                            if scatter_bins[i_bool] == -1:
                                scatter_bins[i_bool] = i

        decay = np.linspace(1.1, 1.15, len(values_to_scatter))
        for i, value_to_scatter in enumerate(values_to_scatter):
            if i < len(labels):
                ax1.scatter(x=value_to_scatter, y=hist_plt[scatter_bins[i]] * decay[i], marker=scatter_shapes[i],
                            color=colors[i], s=60, zorder=20, label=labels[i])
            else:
                ax1.scatter(x=value_to_scatter, y=hist_plt[scatter_bins[i]] * decay[i], marker=scatter_shapes[i],
                            color=colors[i], s=60, zorder=20)
    ax1.legend()

    if tight_x_range:
        ax1.set_xlim(min_range, max_range)
    else:
        ax1.set_xlim(0, 100)
        xticks = np.arange(0, 110, 10)

        ax1.set_xticks(xticks)
        # sce clusters labels
        ax1.set_xticklabels(xticks)
    ax1.yaxis.set_tick_params(labelsize=ticks_label_size)
    ax1.xaxis.set_tick_params(labelsize=ticks_label_size)
    ax1.tick_params(axis='y', colors=axis_labels_color)
    ax1.tick_params(axis='x', colors=axis_labels_color)
    # TO remove the ticks but not the labels
    # ax1.xaxis.set_ticks_position('none')

    if ylabel is None:
        ax1.set_ylabel("Distribution (%)", fontsize=axis_label_font_size, labelpad=20, fontweight=fontweight,
                       fontfamily=fontfamily)
    else:
        ax1.set_ylabel(ylabel, fontsize=axis_label_font_size, labelpad=20, fontweight=fontweight, fontfamily=fontfamily)
    ax1.set_xlabel(xlabel, fontsize=axis_label_font_size, labelpad=20, fontweight=fontweight, fontfamily=fontfamily)

    ax1.xaxis.label.set_color(axis_labels_color)
    ax1.yaxis.label.set_color(axis_labels_color)

    if ax_to_use is None:
        # padding between ticks label and  label axis
        # ax1.tick_params(axis='both', which='major', pad=15)
        fig.tight_layout()
        if save_figure and (path_results is not None):
            # transforming a string in a list
            if isinstance(save_formats, str):
                save_formats = [save_formats]
            time_str = ""
            if with_timestamp_in_file_name:
                time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
            for save_format in save_formats:
                if not with_timestamp_in_file_name:
                    fig.savefig(os.path.join(f'{path_results}', f'{filename}.{save_format}'),
                                format=f"{save_format}",
                                facecolor=fig.get_facecolor())
                else:
                    fig.savefig(os.path.join(f'{path_results}', f'{filename}{time_str}.{save_format}'),
                                format=f"{save_format}",
                                facecolor=fig.get_facecolor())
        plt.close()


def plot_violin_distribution(distribution_data,
                             filename=None,
                             hue=None,
                             order=None,
                             hue_order=None,
                             scale='area',
                             scale_hue=True,
                             gridsize=100,
                             width=0.8,
                             inner=None,
                             split=False,
                             dodge=True,
                             linewidth=None,
                             specify_ins=False,
                             cell_type=None,
                             palette_ins=None,
                             scale_them_all=False,
                             violin_color="white",
                             violin_edgecolor="white",
                             background_color="black",
                             axis_labels_color="white",
                             axis_color="white",
                             axis_label_font_size=20,
                             ticks_label_size=14,
                             ticks_labels_color="white",
                             xlabel=None,
                             ylabel=None,
                             fontweight=None,
                             fontfamily=None,
                             size_fig=None,
                             dpi=100,
                             path_results=None,
                             save_figure=False,
                             save_formats="pdf",
                             with_timestamp_in_file_name=True,
                             ax_to_use=None,
                             max_value=None):
    distribution = np.array(distribution_data)
    ncells = len(distribution)

    if ax_to_use is None:
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                gridspec_kw={'height_ratios': [1]},
                                figsize=size_fig, dpi=dpi)
        ax1.set_facecolor(background_color)
        fig.patch.set_facecolor(background_color)
    else:
        ax1 = ax_to_use

    if specify_ins is True:
        hue = "Cell Type"
        palette = palette_ins
        split = True
        order = ["Pyramidal", "Interneuron"]
    if specify_ins is False:
        hue = None
        palette = None
        split = False
        order = None

    df_subset = pd.DataFrame()
    df_subset['Distribution'] = distribution
    df_subset['Xlabel'] = xlabel
    df_subset['Cell Type'] = cell_type
    svm = sns.violinplot(data=df_subset,
                         x="Xlabel",
                         y="Distribution",
                         hue=hue,
                         order=None,
                         hue_order=order,
                         bw='scott',
                         cut=0,
                         scale='area',
                         scale_hue=True,
                         gridsize=100,
                         width=0.8,
                         inner=inner,
                         split=split,
                         dodge=True,
                         orient=None,
                         linewidth=None,
                         color=violin_color,
                         palette=palette,
                         saturation=0.75,
                         ax=ax1)

    if specify_ins is False:
        svm.collections[0].set_edgecolor(violin_edgecolor)

    svm.set_xticklabels('')
    svm.set_xlabel('')
    svm.set_ylabel('')

    scale_them_all = scale_them_all
    if scale_them_all is True:
        ax1.set_ylim([0, max_value])

    ax1.set_ylabel(ylabel, fontsize=axis_label_font_size, labelpad=20, fontweight=fontweight, fontfamily=fontfamily)
    ax1.set_xlabel(xlabel, fontsize=axis_label_font_size, labelpad=20, fontweight=fontweight, fontfamily=fontfamily)
    ax1.xaxis.label.set_color(axis_labels_color)
    ax1.yaxis.label.set_color(axis_labels_color)
    ax1.spines['left'].set_color(axis_color)
    ax1.spines['right'].set_color(background_color)
    ax1.spines['bottom'].set_color(background_color)
    ax1.spines['top'].set_color(background_color)
    ax1.yaxis.set_tick_params(labelsize=ticks_label_size)
    ax1.xaxis.set_tick_params(labelsize=ticks_label_size)
    ax1.tick_params(axis='y', colors=ticks_labels_color)
    ax1.tick_params(axis='x', colors=background_color)

    if ax_to_use is None:
        # padding between ticks label and  label axis
        # ax1.tick_params(axis='both', which='major', pad=15)
        fig.tight_layout()
        if save_figure and (path_results is not None):
            # transforming a string in a list
            if isinstance(save_formats, str):
                save_formats = [save_formats]
            time_str = ""
            if with_timestamp_in_file_name:
                time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
            for save_format in save_formats:
                if not with_timestamp_in_file_name:
                    fig.savefig(os.path.join(f'{path_results}', f'{filename}.{save_format}'),
                                format=f"{save_format}",
                                facecolor=fig.get_facecolor())
                else:
                    fig.savefig(os.path.join(f'{path_results}', f'{filename}{time_str}.{save_format}'),
                                format=f"{save_format}",
                                facecolor=fig.get_facecolor())
        plt.close()


def plot_multiple_violin_distributions(distribution_data,
                                       filename=None,
                                       hue=None,
                                       order=None,
                                       hue_order=None,
                                       scale='area',
                                       scale_hue=True,
                                       gridsize=100,
                                       width=0.8,
                                       inner=None,
                                       split=False,
                                       dodge=True,
                                       linewidth=None,
                                       specify_ins=False,
                                       cell_type=None,
                                       palette_ins=None,
                                       scale_them_all=False,
                                       violin_color="white",
                                       violin_edgecolor="white",
                                       background_color="black",
                                       axis_labels_color="white",
                                       axis_color="white",
                                       axis_label_font_size=15,
                                       ticks_label_size=8,
                                       ticks_labels_color="white",
                                       xlabel=None,
                                       ylabel=None,
                                       fontweight=None,
                                       fontfamily=None,
                                       size_fig=None,
                                       dpi=100,
                                       path_results=None,
                                       save_figure=False,
                                       save_formats="pdf",
                                       with_timestamp_in_file_name=True):
    distribution_list = distribution_data
    n_sessions = len(distribution_list)
    cell_type_lists = cell_type
    id_list = xlabel
    n_violin_plots = len(distribution_list)
    print(f"Plotting {n_violin_plots} violin-plots ")
    n_plots = n_violin_plots
    ins_spec = np.zeros(len(distribution_list))
    if isinstance(specify_ins, list):
        for session in range(n_sessions):
            if specify_ins[session] is True:
                ins_spec[session] = 1

    max_values = []
    for distribution_index in range(n_plots):
        max_values.append(np.nanmax(distribution_list[distribution_index]))
    max_value = np.nanmax(max_values)

    if n_plots > 6:
        max_n_lines = 5
    else:
        max_n_lines = 2
    n_lines = n_plots if n_plots <= max_n_lines else max_n_lines
    n_col = math.ceil(n_plots / n_lines)
    print(f"n_lines {n_lines}, n_cols {n_col}")
    fig, axes = plt.subplots(nrows=n_lines, ncols=n_col,
                             gridspec_kw={'width_ratios': [1] * n_col, 'height_ratios': [1] * n_lines},
                             figsize=size_fig)
    fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    fig.patch.set_facecolor(background_color)

    if n_lines + n_col == 2:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax_index, ax in enumerate(axes):
        ax.set_facecolor(background_color)
        ax.spines['left'].set_color(background_color)
        ax.spines['right'].set_color(background_color)
        ax.spines['bottom'].set_color(background_color)
        ax.spines['top'].set_color(background_color)
        ax.tick_params(axis='y', colors=background_color)
        ax.tick_params(axis='x', colors=background_color)
        if ax_index >= n_violin_plots:
            continue
        if ins_spec[ax_index] == 1:
            specify_ins = True
        if ins_spec[ax_index] == 0:
            specify_ins = False

        distribution = distribution_list[ax_index]
        if cell_type is not None:
            cell_type = cell_type_lists[ax_index]

        xlabel = id_list[ax_index]
        print(f"Plotting {xlabel}: specify interneurons is {specify_ins}")
        plot_violin_distribution(distribution,
                                 filename=filename,
                                 hue=None,
                                 order=None,
                                 hue_order=None,
                                 scale='area',
                                 scale_hue=True,
                                 gridsize=100,
                                 width=0.8,
                                 inner=inner,
                                 split=False,
                                 dodge=True,
                                 linewidth=None,
                                 specify_ins=specify_ins,
                                 cell_type=cell_type,
                                 palette_ins=palette_ins,
                                 scale_them_all=scale_them_all,
                                 violin_color=violin_color,
                                 violin_edgecolor=violin_edgecolor,
                                 background_color=background_color,
                                 axis_labels_color=axis_labels_color,
                                 axis_color=axis_color,
                                 axis_label_font_size=axis_label_font_size,
                                 ticks_label_size=ticks_label_size,
                                 ticks_labels_color=ticks_labels_color,
                                 xlabel=xlabel,
                                 ylabel=ylabel,
                                 fontweight=fontweight,
                                 fontfamily=fontfamily,
                                 size_fig=None,
                                 dpi=dpi,
                                 path_results=None,
                                 save_figure=False,
                                 save_formats="pdf",
                                 with_timestamp_in_file_name=True,
                                 ax_to_use=ax,
                                 max_value=max_value)

    if save_figure and (path_results is not None):
        # transforming a string in a list
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        time_str = ""
        if with_timestamp_in_file_name:
            time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        for save_format in save_formats:
            if not with_timestamp_in_file_name:
                fig.savefig(os.path.join(f'{path_results}', f'{filename}.{save_format}'),
                            format=f"{save_format}",
                            facecolor=fig.get_facecolor())
            else:
                fig.savefig(os.path.join(f'{path_results}', f'{filename}{time_str}.{save_format}'),
                            format=f"{save_format}",
                            facecolor=fig.get_facecolor())
    plt.close()
