# import webcolors

# qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12 + 11 diverting

BREWER_COLORS = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                 '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
                 '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
                 '#74add1', '#4575b4', '#313695']


def rgb_to_name(rgb_triplet, with_float_values):
    """

    Args:
        rgb_triplet:
        with_float_values: bool, if True, means the triplets value are between 0 and 1, otherwise int between 1 and 255

    Returns:

    """
    if with_float_values:
        rgb_triplet = [int(round(c*255)) for c in rgb_triplet]
        if len(rgb_triplet) > 3:
            # removing alpha if there
            rgb_triplet = rgb_triplet[:3]
    try:
        return webcolors.rgb_to_name(rgb_triplet=rgb_triplet)
    except ValueError:
        return str(rgb_triplet)