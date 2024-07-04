import os
from .tools import make_dir

def wget_file(url, dest_dir, overwrite=True):
    """Download file from a given url using wget.
    Output folder will be created if not exist.

    Parameters
    ----------
    url : str
        url to download resources
    dest_dir : str
        path to output folder
    overwite : bool
        whether to overwrite file if already exists
    """
    try:
        import wget
        make_dir(dest_dir)
        fname = url.split('/')[-1]
        fpath = os.path.join(dest_dir, fname)
        if os.path.exists(fpath) and (not overwrite):
            print(f"{fpath} already exists. Do not overwrite.")
        else:
            wget.download(url, out=dest_dir)
            print(f"SUCCEEDED downloading {url} to {fpath}.")
    except Exception as e:
        print(f"FAILED to download {url}. REASON: {e}")

def os_wget_file(url, dest_dir, overwrite=True):
    """Download file from a given url using operating system wget.
    Output folder will be created if not exist.

    Parameters
    ----------
    url : str
        url to download resources
    dest_dir : str
        path to output folder
    overwite : bool
        whether to overwrite file if already exists
    """
    try:
        make_dir(dest_dir)
        fname = url.split('/')[-1]
        fpath = os.path.join(dest_dir, fname)
        if os.path.exists(fpath) and (not overwrite):
            print(f"{fpath} already exists. Do not overwrite.")
        else:
            os.system(f"""wget "{url}" -O "{fpath}" """)
            print(f"SUCCEEDED downloading {url} to {fpath}.")
    except Exception as e:
        print(f"FAILED to download {url}. REASON: {e}")
