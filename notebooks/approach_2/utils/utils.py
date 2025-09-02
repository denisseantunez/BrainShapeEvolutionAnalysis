import numpy as np

#  a way to track the current torch device globally
global_config = {
    'device': None,
}


def get_device():
    """
    Return the global torch device
    """
    return global_config.get('device')


def set_device(device):
    """
    Set the global torch device
    """
    global_config['device'] = device


def read_file_list(filename, prefix=None, suffix=None):
    """
    Reads a list of files from a line-seperated text file
    """
    with open(filename, 'r') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content if x.strip()]
    if prefix is not None:
        filelist = [prefix + f for f in filelist]
    if suffix is not None:
        filelist = [f + suffix for f in filelist]
    return filelist


def flatten_lists(lst):
    """                                                                         
    return a list of elements that are not lists                                
    if lst is not a list, will return [lst]                                     
    otherwise, will flatten any hierarchy to a long list of lists               
    """
    if not isinstance(lst, (tuple, list)):
        lst = [lst]

    long_lst = []
    for f in lst:
        if isinstance(f, (tuple, list)):
            long_lst.extend(flatten_lists(f))
        else:
            long_lst.append(f)
    return long_lst

def rebase_labels(labels):
    '''Rebase labels and return lookup table (LUT) to convert to new labels in
    interval [0, N[ as: LUT[label_map]. Be sure to pass all possible labels.'''
    labels = np.unique(labels) # Sorted.
    assert np.issubdtype(labels.dtype, np.integer), 'non-integer data'
    lab_to_ind = np.zeros(np.max(labels) + 1, dtype='int_')
    for i, lab in enumerate(labels):
        lab_to_ind[ lab ] = i
    ind_to_lab = labels
    return lab_to_ind, ind_to_lab    