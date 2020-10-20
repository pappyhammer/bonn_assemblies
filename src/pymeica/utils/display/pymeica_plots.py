import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
from datetime import datetime
import seaborn as sns
import math


def plot_scatter_family(data_dict, colors_dict,
                        filename,
                        y_label,
                        label_to_legend=None,
                        marker_to_legend=None,
                        path_results=None, y_lim=None,
                        x_label=None,
                        y_log=False,
                        scatter_size=200,
                        scatter_alpha=1,
                        background_color="black",
                        lines_plot_values=None,
                        plots_linewidth=2,
                        link_scatter=False,
                        labels_color="white",
                        with_x_jitter=0.2,
                        with_y_jitter=None,
                        x_labels_rotation=None,
                        h_lines_y_values=None,
                        with_text=False,
                        default_marker='o',
                        text_size=5,
                        save_formats="pdf",
                        dpi=200,
                        with_timestamp_in_file_name=True):
    """
    Plot family of scatters (same color and label) with possibly lines that are associated to it.
    :param data_dict: key is a label, value is a list of up to 4 (2 mandatory) list of int, of same size,
    first one are the x-value,
    second one is the y-values
    Third one is a text to write in the scatter
    Fourth one is the marker to display, could be absent, default_marker will be used
    Fifth one is the number of elements that allows to get this number (like number
    of sessions)
    :param colors_dict = key is a label, value is a color
    :param label_to_legend: (dict) if not None, key are label of data_dict, value is the label to be displayed as legend
    :param marker_to_legend: (dict) if not None, key are marker of data_dict, value is the label to be displayed as legend
    :param filename:
    :param lines_plot_values: (dict) same keys as data_dict, value is a list of list of 2 list of int or float,
    representing x & y value of plot to trace
    :param y_label:
    :param y_lim: tuple of int,
    :param link_scatter: draw a line between scatters
    :param save_formats:
    :param with_text: if True and data available in data_dict, then text is plot in the scatter
    :return:
    """
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(12, 12), dpi=dpi)

    ax1.set_facecolor(background_color)

    fig.patch.set_facecolor(background_color)

    # labels = []
    # data_list = []
    # scatter_text_list = []
    # medians_values = []
    min_x_value = 10000
    max_x_value = 0

    for label, data_to_scatters in data_dict.items():
        if len(data_to_scatters[0]) == 0:
            continue
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
        if len(data_to_scatters) > 3:
            markers = data_to_scatters[3]
            for index in range(len(x_pos)):
                # if too slow, see to regroup by scatter value
                ax1.scatter(x_pos[index], y_pos[index],
                            color=colors_dict[label],
                            alpha=scatter_alpha,
                            marker=markers[index],
                            edgecolors=labels_color,
                            # label=label_legend,
                            s=scatter_size, zorder=21)
        else:
            ax1.scatter(x_pos, y_pos,
                        color=colors_dict[label],
                        alpha=scatter_alpha,
                        marker=default_marker,
                        edgecolors=labels_color,
                        # label=label_legend,
                        s=scatter_size, zorder=21)

        if with_text and len(data_to_scatters) > 2:
            # then the third dimension is a text to plot in the scatter
            for scatter_index in np.arange(len(data_to_scatters[2])):
                scatter_text = str(data_to_scatters[2][scatter_index])
                if len(scatter_text) > 3:
                    font_size = text_size - 2
                else:
                    font_size = text_size
                ax1.text(x=x_pos[scatter_index], y=y_pos[scatter_index],
                         s=scatter_text, color="black", zorder=50,
                         ha='center', va="center", fontsize=font_size, fontweight='bold')

        if link_scatter:
            ax1.plot(x_pos, y_pos, zorder=30, color=colors_dict[label],
                     linewidth=plots_linewidth)

        if lines_plot_values is not None:
            if label in lines_plot_values:
                for lines_coordinates in lines_plot_values[label]:
                    x_pos, y_pos = lines_coordinates
                    ax1.plot(x_pos, y_pos, zorder=35, color=colors_dict[label],
                             linewidth=plots_linewidth)

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

    legend_elements = []
    for label, color in colors_dict.items():
        if label not in label_to_legend:
            continue
        label_legend = label_to_legend[label] if label_to_legend is not None else label
        legend_elements.append(Patch(facecolor=color,
                                     edgecolor='black', label=label_legend))
    if (marker_to_legend is not None) and (len(marker_to_legend) > 0):
        for marker, marker_legend in marker_to_legend.items():
            legend_elements.append(Line2D([0], [0], marker=marker, color="w", lw=0, label=marker_legend,
                                          markerfacecolor='black', markersize=12))
    ax1.legend(handles=legend_elements)

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


def plot_ca_param_over_night_by_sleep_stage(subjects_data, side_to_analyse, param_name,
                                            fct_to_get_param,
                                            y_axis_label,
                                            color_by_sleep_stage_dict,
                                            only_ca_with_ri, min_repeat_in_ca, brain_region_to_marker,
                                            marker_to_brain_region,
                                            with_text,
                                            n_ru_in_text,
                                            ratio_ru_in_text,
                                            sleep_stages_to_analyse_by_subject, results_path,
                                            save_formats, dpi,
                                            referenced_ca=None,
                                            min_n_cells_assemblies=2,
                                            min_cells_in_cell_assemblies=2,
                                            hyponogram_y_step=0.25,
                                            plots_linewidth=1,
                                            individual_plot_for_ss=True,
                                            with_mean_lines=False,
                                            from_first_stage_available=True,
                                            h_lines_y_values=None):
    """
    Allows to plot a parameter linked to cell assemblies in order to plot its evolution (in y) over the course
    of the night (in x)
    :param subjects_data:
    :param side_to_analyse: (str)value 'L', 'R', or 'L&R'
    :param param_name: (str) name description used for naming file,
    :param fct_to_get_param: (function) take in argument an instance of CellAssembly and return a value, that correspond
    to the CA param which is evolution will be ploted
    :param y_axis_label
    :param min_n_cells_assemblies: min n_cell_assemblies in mcad_outcome so we consider inserting this CA in the
    plot
    :param min_cells_in_cell_assemblies:
    :param sleep_stages_to_analyse_by_subject:
    :param referenced_ca: if not None, CellAssembly instance that will be used to compare two instances
    :param individual_plot_for_ss: if True, a plot if made for each sleep_stage
    :param results_path:
    :param save_formats:
    :param h_lines_y_values: list of float, if not None, a horizontal line will be plot for each value (y)
    :param hyponogram_y_step: (float) how much space to let between 2 lines of the hyponogram
    :return:
    """
    # print("plot_ca_proportion_night_by_sleep_stage")

    avg_fct = np.mean
    subject_descr = ""
    # key is sleep_stage_name, value is a list of 2 list, first list contains the time elapsed since falling asleep,
    # second is the param of the assembly (up to 4 list are the same length)
    ca_param_by_stage_dict = dict()
    time_by_stage_with_ca_dict = dict()
    time_total_by_stage_dict = dict()
    # key is sleep_stage name, value number of CA
    n_assemblies_by_stage_dict = dict()
    # key is sleep_stage name, value list of values
    all_values_by_sleep_stage = dict()
    # key is brain region, value number of CA
    count_ca_by_brain_region = dict()
    # key is brain region, value list of values
    all_values_by_brain_region = dict()

    for subject_data in subjects_data:
        subject_id = subject_data.identifier
        subject_descr = subject_descr + subject_id + "_"
        if side_to_analyse == "L&R":
            sides = ['L', 'R']
        else:
            sides = [side_to_analyse]
        for side in sides:
            # print(f'plot_ca_proportion_night_by_sleep_stage side {side}')
            for sleep_stage_index in np.sort(sleep_stages_to_analyse_by_subject[subject_id]):
                sleep_stage = subject_data.sleep_stages[sleep_stage_index]
                if sleep_stage.sleep_stage not in time_total_by_stage_dict:
                    time_total_by_stage_dict[sleep_stage.sleep_stage] = 0
                    time_by_stage_with_ca_dict[sleep_stage.sleep_stage] = 0
                time_total_by_stage_dict[sleep_stage.sleep_stage] += sleep_stage.duration_sec
                elapsed_time = subject_data.elapsed_time_from_falling_asleep(sleep_stage=sleep_stage,
                                                                             from_first_stage_available=
                                                                             from_first_stage_available)
                if elapsed_time < 0:
                    continue
                # sleep_stage.sleep_stage is a string representing the sleep stage like 'W' or '3'

                if len(sleep_stage.mcad_outcomes) == 0:
                    # then there is no cell assembly
                    continue

                time_cover_by_bin_tuples = 0
                # to count the chunks
                current_time_in_sleep_stage = 0
                for bins_tuple, mcad_outcome in sleep_stage.mcad_outcomes.items():
                    n_bins = bins_tuple[1] - bins_tuple[0] + 1
                    chunk_duration_in_sec = (n_bins * mcad_outcome.spike_trains_bin_size) / 1000

                    if mcad_outcome.side != side:
                        # only keeping the outcome from the correct side
                        current_time_in_sleep_stage += chunk_duration_in_sec
                        continue
                    if mcad_outcome.n_cell_assemblies < min_n_cells_assemblies:
                        current_time_in_sleep_stage += chunk_duration_in_sec
                        continue

                    # init
                    if sleep_stage.sleep_stage not in ca_param_by_stage_dict:
                        ca_param_by_stage_dict[sleep_stage.sleep_stage] = [[], [], [], []]
                        n_assemblies_by_stage_dict[sleep_stage.sleep_stage] = 0
                        all_values_by_sleep_stage[sleep_stage.sleep_stage] = []

                    # instances of CellAssembly
                    cell_assembly_added = False
                    for cell_assembly in mcad_outcome.cell_assemblies:
                        if only_ca_with_ri and (cell_assembly.n_responsive_units == 0):
                            continue
                        if cell_assembly.n_repeats < min_repeat_in_ca:
                            continue

                        if cell_assembly.n_units < min_cells_in_cell_assemblies:
                            continue

                        if referenced_ca is not None:
                            param_value = fct_to_get_param(cell_assembly, referenced_ca)
                        else:
                            param_value = fct_to_get_param(cell_assembly)

                        if param_value is None:
                            continue

                        n_assemblies_by_stage_dict[sleep_stage.sleep_stage] += 1

                        if not cell_assembly_added:
                            time_by_stage_with_ca_dict[sleep_stage.sleep_stage] += chunk_duration_in_sec
                            time_elapsed_in_sec = elapsed_time + current_time_in_sleep_stage
                            time_elapsed_in_hours = time_elapsed_in_sec / 3600
                        ca_param_by_stage_dict[sleep_stage.sleep_stage][0].append(time_elapsed_in_hours)
                        # negative log value
                        ca_param_by_stage_dict[sleep_stage.sleep_stage][1].append(param_value)
                        if ratio_ru_in_text:
                            ratio_ri = cell_assembly.n_responsive_units / cell_assembly.n_units
                            ratio_ri = f"{ratio_ri:.2f}"
                            ca_param_by_stage_dict[sleep_stage.sleep_stage][2].append(ratio_ri)
                        elif n_ru_in_text:
                            ca_param_by_stage_dict[sleep_stage.sleep_stage][2].append(cell_assembly.n_responsive_units)

                        # adding a marker representing the main brain region is this assembly
                        brain_region, units_proportion = cell_assembly.main_brain_region
                        marker = brain_region_to_marker[brain_region]
                        ca_param_by_stage_dict[sleep_stage.sleep_stage][3].append(marker)

                        # key is brain region, value number of CA
                        if brain_region not in count_ca_by_brain_region:
                            # key is brain region, value list of values
                            count_ca_by_brain_region[brain_region] = 0
                            all_values_by_brain_region[brain_region] = []
                        count_ca_by_brain_region[brain_region] += 1
                        all_values_by_brain_region[brain_region].append(param_value)
                        all_values_by_sleep_stage[sleep_stage.sleep_stage].append(param_value)

                        cell_assembly_added = True

                    current_time_in_sleep_stage += chunk_duration_in_sec

    # moving winodw to average over time the param
    # key is the sleep stage name, value is a list of list of list of 2 float representing (x, y) for line plots
    avg_param_hypno_by_stage_dict = dict()
    # in hours
    window_length = 0.5
    step_length = 0.25
    min_time = 10000
    max_time = 0
    for sleep_stage_name, scatter_values in ca_param_by_stage_dict.items():
        times = scatter_values[0]
        if len(times) == 0:
            continue
        min_time = min(min_time, np.min(times))
        max_time = max(max_time, np.max(times))
    bin_edges = np.arange(min_time, max_time + step_length, step_length)

    y_pos_sep = -1 * hyponogram_y_step
    y_pos_sleep_stage = y_pos_sep
    for sleep_stage_name, scatter_values in ca_param_by_stage_dict.items():
        avg_param_hypno_by_stage_dict[sleep_stage_name] = [[[], []]]
        if with_mean_lines:
            times = scatter_values[0]
            param_values = np.asarray(scatter_values[1])
            for step_index, bin_edge in enumerate(bin_edges[:-1]):
                next_bin_edge = bin_edges[step_index + 1]
                center_time = (bin_edge + next_bin_edge) / 2
                indices = np.where(np.logical_and(times >= bin_edge, times <= next_bin_edge))[0]
                avg_param_hypno_by_stage_dict[sleep_stage_name][0][0].append(center_time)
                if len(indices) == 0:
                    avg_param_hypno_by_stage_dict[sleep_stage_name][0][1].append(0)
                else:
                    avg_param_value = np.mean(param_values[indices])
                    avg_param_hypno_by_stage_dict[sleep_stage_name][0][1].append(avg_param_value)

        for sleep_stage_index in np.sort(sleep_stages_to_analyse_by_subject[subject_id]):
            sleep_stage = subject_data.sleep_stages[sleep_stage_index]
            if sleep_stage.sleep_stage != sleep_stage_name:
                continue
            elapsed_time = subject_data.elapsed_time_from_falling_asleep(sleep_stage=sleep_stage,
                                                                         from_first_stage_available=
                                                                         from_first_stage_available)
            if elapsed_time < 0:
                continue
            new_epoch = [[elapsed_time / 3600, (elapsed_time + sleep_stage.duration_sec) / 3600],
                         [y_pos_sleep_stage, y_pos_sleep_stage]]
            avg_param_hypno_by_stage_dict[sleep_stage_name].append(new_epoch)
        y_pos_sleep_stage += y_pos_sep

    # legend by brain region
    brain_region_legend_dict = dict()
    brain_region_legend_dict.update(marker_to_brain_region)
    for brain_region in count_ca_by_brain_region.keys():
        marker = brain_region_to_marker[brain_region]
        legend = brain_region_legend_dict[marker]
        count_ca = count_ca_by_brain_region[brain_region]
        avg_value = avg_fct(all_values_by_brain_region[brain_region])
        new_legend = legend + f" (x{count_ca} CA, avg={avg_value:.1f})"
        brain_region_legend_dict[marker] = new_legend

    # h_lines_y_values = [-1 * np.log(0.05)]
    # for sleep_stage, duration_in_stage in total_sleep_duration_by_stage.items():
    if individual_plot_for_ss:
        for sleep_stage_name, scatter_values in ca_param_by_stage_dict.items():
            data_dict = {sleep_stage_name: scatter_values}
            ca_time_sec = time_by_stage_with_ca_dict[sleep_stage_name]
            total_time_sec = time_total_by_stage_dict[sleep_stage_name]
            ratio_ca_time_total_time = (ca_time_sec / total_time_sec) * 100
            avg_value_ss = avg_fct(all_values_by_sleep_stage[sleep_stage_name])
            legend = f"(x{n_assemblies_by_stage_dict[sleep_stage_name]} CA, avg={avg_value_ss:.1f}) " \
                     f"{ratio_ca_time_total_time:.1f}% of {(total_time_sec / 60):.1f} min"
            label_to_legend_dict = {sleep_stage_name: f"{sleep_stage_name}: {legend}"}

            avg_param_dict = {sleep_stage_name: avg_param_hypno_by_stage_dict[sleep_stage_name]}

            plot_scatter_family(data_dict=data_dict,
                                label_to_legend=label_to_legend_dict,
                                colors_dict=color_by_sleep_stage_dict,
                                filename=f"{subject_descr}{param_name}_ca_over_night_stage_"
                                         f"{sleep_stage_name}_{side_to_analyse}",
                                y_label=y_axis_label,
                                path_results=results_path,  # y_lim=[0, 100],
                                x_label="Time (hours)",
                                y_log=False,
                                h_lines_y_values=h_lines_y_values,
                                scatter_size=150,
                                scatter_alpha=0.8,
                                plots_linewidth=plots_linewidth,
                                lines_plot_values=avg_param_dict,
                                background_color="black",
                                link_scatter=False,
                                labels_color="white",
                                with_x_jitter=0.05,
                                with_y_jitter=None,
                                x_labels_rotation=None,
                                marker_to_legend=brain_region_legend_dict,
                                with_text=with_text,
                                text_size=5,
                                save_formats=save_formats,
                                dpi=dpi,
                                with_timestamp_in_file_name=True)

    label_to_legend_dict = dict()
    for sleep_stage_name in ca_param_by_stage_dict.keys():
        ca_time_sec = time_by_stage_with_ca_dict[sleep_stage_name]
        total_time_sec = time_total_by_stage_dict[sleep_stage_name]
        ratio_ca_time_total_time = (ca_time_sec / total_time_sec) * 100
        avg_value_ss = avg_fct(all_values_by_sleep_stage[sleep_stage_name])
        # time in CA from this stage in min: (ca_time_sec / 60)
        legend = f"(x{n_assemblies_by_stage_dict[sleep_stage_name]} CA, avg={avg_value_ss:.1f}) " \
                 f"{ratio_ca_time_total_time:.1f}% of {(total_time_sec / 60):.1f} min"
        label_to_legend_dict[sleep_stage_name] = f"{sleep_stage_name}: {legend}"

    # TODO: See to add option to have a different shape for a cell assembly depending of if it contains RU
    plot_scatter_family(data_dict=ca_param_by_stage_dict,
                        label_to_legend=label_to_legend_dict,
                        colors_dict=color_by_sleep_stage_dict,
                        filename=f"{subject_descr}{param_name}_ca_over_night_{side_to_analyse}",
                        y_label=y_axis_label,
                        path_results=results_path,  # y_lim=[0, 100],
                        x_label="Time (hours)",
                        y_log=False,
                        h_lines_y_values=h_lines_y_values,
                        scatter_size=150,
                        scatter_alpha=0.8,
                        plots_linewidth=plots_linewidth,
                        lines_plot_values=avg_param_hypno_by_stage_dict,
                        background_color="black",
                        link_scatter=False,
                        marker_to_legend=brain_region_legend_dict,
                        labels_color="white",
                        with_x_jitter=0.05,
                        with_y_jitter=None,
                        with_text=with_text,
                        x_labels_rotation=None,
                        save_formats=save_formats,
                        dpi=dpi,
                        with_timestamp_in_file_name=True)


def plot_heatmap(heatmap_matrix, file_name, path_results, y_ticks_labels=None, x_ticks_labels=None,
                                                 x_ticks_pos=None, save_formats=["png"]):
    # src: https://www.geodose.com/2018/01/creating-heatmap-in-python-from-scratch.html
    # DEFINE GRID SIZE AND RADIUS(h)
    grid_size = 1
    h = 5

    x = []
    y = []
    for stim_value in np.arange(heatmap_matrix.shape[0]):
        for time_bin in np.where(heatmap_matrix[stim_value])[0]:
            for n in np.arange(heatmap_matrix[stim_value, time_bin]):
                x.append(time_bin)
                y.append(stim_value)
    if len(x) == 0:
        print(f"Empty heatmap_matrix: {np.sum(heatmap_matrix)}")
        return

    x = np.array(x)
    y = np.array(y)
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)

    # CONSTRUCT GRID
    x_grid = np.arange(x_min - h, x_max + h, grid_size)
    y_grid = np.arange(y_min - h, y_max + h, grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    # GRID CENTER POINT
    xc = x_mesh + (grid_size / 2)
    yc = y_mesh + (grid_size / 2)

    # FUNCTION TO CALCULATE INTENSITY WITH QUARTIC KERNEL
    def kde_quartic(d, h):
        dn = d / h
        P = (15 / 16) * (1 - dn ** 2) ** 2
        return P

    # PROCESSING
    intensity_list = []
    for j in range(len(xc)):
        intensity_row = []
        for k in range(len(xc[0])):
            kde_value_list = []
            for i in range(len(x)):
                # CALCULATE DISTANCE
                d = math.sqrt((xc[j][k] - x[i]) ** 2 + (yc[j][k] - y[i]) ** 2)
                if d <= h:
                    p = kde_quartic(d, h)
                else:
                    p = 0
                kde_value_list.append(p)
            # SUM ALL INTENSITY VALUE
            p_total = sum(kde_value_list)
            intensity_row.append(p_total)
        intensity_list.append(intensity_row)

    intensity = np.array(intensity_list)

    background_color = "black"
    fig, ax = plt.subplots()
    plt.pcolormesh(x_mesh, y_mesh, intensity)
    plt.plot(x, y, '|', color="red")
    plt.colorbar()

    fig.tight_layout()
    # adjust the space between axis and the edge of the figure
    # https://matplotlib.org/faq/howto_faq.html#move-the-edge-of-an-axes-to-make-room-for-tick-labels
    # fig.subplots_adjust(left=0.2)
    ax.set_facecolor(background_color)

    labels_color = "white"
    ax.xaxis.label.set_color(labels_color)
    ax.yaxis.label.set_color(labels_color)

    # ax.yaxis.set_tick_params(labelsize=20)
    # ax.xaxis.set_tick_params(labelsize=20)
    ax.tick_params(axis='y', colors=labels_color)
    ax.tick_params(axis='x', colors=labels_color)
    # if ticks_labels is not None:
    #     ax.set_xticklabels(ticks_labels)
    #     ax.set_yticklabels(ticks_labels)
    if y_ticks_labels is not None:
        ax.set_yticks(np.arange(0, len(y_ticks_labels)))
        ax.set_yticklabels(y_ticks_labels)

    if x_ticks_labels is not None:
        ax.set_xticks(x_ticks_pos)
        ax.set_xticklabels(x_ticks_labels)

    ax.set_ylim(-1, len(y_ticks_labels))

    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(colors=labels_color)

    fig.patch.set_facecolor(background_color)

    if isinstance(save_formats, str):
        save_formats = [save_formats]

    for save_format in save_formats:
        fig.savefig(f'{path_results}/{file_name}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())
    plt.close()


def plot_transition_heatmap(heatmap_content, annot,
                            file_name, results_path, y_ticks_labels=None, x_ticks_labels=None, x_ticks_pos=None,
                            y_ticks_pos=None, save_formats="png", dpi=300):
    """
        Plot a transition matrix heatmap
        :param heatmap_content: a nxn array, containing the value that will be color coded
        :param annot: a nxn array, containing the value that will be displayed in each case
        :param file_name:
        :param path_results:
        :param y_ticks_labels:
        :param x_ticks_labels:
        :param x_ticks_pos:
        :param y_ticks_pos:
        :return:
    """
    background_color = "black"
    fig, ax = plt.subplots(dpi=dpi)
    # fmt='d' means integers
    ax = sns.heatmap(heatmap_content, annot=annot, fmt='d', cmap="YlGnBu",
                     vmin=np.min(heatmap_content), vmax=np.max(heatmap_content))
    # vmin=0, vmax=100
    # adjust the space between axis and the edge of the figure
    # https://matplotlib.org/faq/howto_faq.html#move-the-edge-of-an-axes-to-make-room-for-tick-labels
    # fig.subplots_adjust(left=0.2)
    ax.set_facecolor(background_color)

    labels_color = "white"
    ax.xaxis.label.set_color(labels_color)
    ax.yaxis.label.set_color(labels_color)

    # ax.yaxis.set_tick_params(labelsize=20)
    # ax.xaxis.set_tick_params(labelsize=20)
    ax.tick_params(axis='y', colors=labels_color, labelsize=5)
    ax.tick_params(axis='x', colors=labels_color, labelsize=5)
    if x_ticks_labels is not None:
        if x_ticks_pos is not None:
            ax.set_xticks(x_ticks_pos)
        ax.set_xticklabels(x_ticks_labels)
    if y_ticks_labels is not None:
        if y_ticks_pos is not None:
            ax.set_yticks(y_ticks_pos)
        ax.set_yticklabels(y_ticks_labels)

    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(colors=labels_color, labelsize=8)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    for tick in ax.get_yticklabels():
        tick.set_rotation(45)

    fig.tight_layout()

    fig.patch.set_facecolor(background_color)
    if isinstance(save_formats, str):
        save_formats = [save_formats]

    for save_format in save_formats:
        fig.savefig(f'{results_path}/{file_name}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())
    plt.close()
