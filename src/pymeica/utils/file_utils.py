import os


def find_files(dir_to_explore, keywords=None, extensions=("yaml", "yml")):
    """
    Recursive function that will go through all subdirectories in dir_to_explore
    and look for files that contain keywords with given extensions
    Args:
        dir_to_explore:

        keywords: None or list of str. strings representing the keywords the file could contain

        extensions: extensions of the files we're looking for

    Returns:

    """
    files_found = []
    for (dir_path, dir_names, local_filenames) in os.walk(dir_to_explore):

        for file_name in local_filenames:
            valid_extension = False
            for extension in extensions:
                if file_name.endswith(extension):
                    valid_extension = True
                    continue
            if not valid_extension:
                continue
            valid_keyword = False
            if keywords is not None:
                for keyword in keywords:
                    if keyword in file_name:
                        valid_keyword = True
                        continue
            else:
                valid_keyword = True

            if valid_keyword:
                files_found.append(os.path.join(dir_path, file_name))

    return files_found