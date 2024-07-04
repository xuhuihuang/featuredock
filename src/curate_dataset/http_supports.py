"""
This script contains functions for http requests.
"""

import os
import urllib
import time
import requests
import json
import pickle

def download_single_file(url, dest, overwrite=True):
    if os.path.exists(dest) and not overwrite:
        print(f"File: {dest} already exists.")
    else:
        try:
            urllib.request.urlretrieve(url, dest)
        except:
            print(f'Fail to download {url} to {dest}.')


def request_limited(url, rtype="GET", num_attempts=3, \
        sleep_time=0.5, **kwargs):
    """
    This function is copied from pypdb
    (https://github.com/williamgilpin/pypdb).
    HTML request with rate-limiting base on response code.

    Parameters
    ----------
    url : str
        The url for the request
    rtype : str
        The request type (oneof ["GET", "POST"])
    num_attempts : int
        In case of a failed retrieval, the number of attempts to try again
    sleep_time : int
        The amount of time to wait between requests, in case of
        API rate limits
    **kwargs : dict
        The keyword arguments to pass to the request

    Returns
    -------
    response : requests.models.Response
        The server response object. Only returned if request was successful,
        otherwise returns None.
    """

    if rtype not in ["GET", "POST"]:
        print("Request type not recognized")
        return None

    total_attempts = 0
    while total_attempts <= num_attempts:
        if rtype == "GET":
            response = requests.get(url, **kwargs)
        elif rtype == "POST":
            response = requests.post(url, **kwargs)

        if response.status_code == 200:
            return response
        if response.status_code == 429:
            curr_sleep = (1 + total_attempts) * sleep_time
            print("Too many requests, waiting " + str(curr_sleep) + " s")
            time.sleep(curr_sleep)
        elif 500 <= response.status_code < 600:
            print("Server error encountered. Retrying")
        total_attempts += 1
    print("Too many failures on requests. Exiting...")
    return None


def download_pdb_file(pdbid, dest, zipped=False, overwrite=True):
    """
    Download single pdb file from RCSB PDB.
    """
    if zipped:
        root_url = 'https://files.rcsb.org/download/{}.pdb.gz'
    else:
        root_url = 'https://files.rcsb.org/download/{}.pdb'
    url = root_url.format(pdbid.strip().upper())
    download_single_file(url, dest, overwrite)


def download_pdb1_file(pdbid, dest, zipped=False, overwrite=True):
    """
    Download biological assembly from RCSB PDB.
    """
    if zipped:
        root_url = 'https://files.rcsb.org/download/{}.pdb1.gz'
    else:
        root_url = 'https://files.rcsb.org/download/{}.pdb1'
    url = root_url.format(pdbid.strip().upper())
    download_single_file(url, dest, overwrite)


def get_info(pdb_id, filename, store=True, overwrite=False):
    '''
    This function is derived from pypdb (https://github.com/williamgilpin/pypdb).

    Look up all information about a given PDB ID 
    Parameters
    ----------
    pdb_id : string
        A 4 character string giving a pdb entry of interest
    url_root : string
        The string root of the specific url for the request type
    Returns
    -------
    out : json
        An ordered dictionary object corresponding to entry information
    '''
    url_root = 'https://data.rcsb.org/rest/v1/core/entry/{}'
    if os.path.exists(filename) and not overwrite and store:
        print('%s already exists.' % filename)
        return None

    pdb_id = pdb_id.upper()
    url = url_root.format(pdb_id)
    response = request_limited(url)

    if response is None or response.status_code != 200:
        print("%s: Retrieval failed, returning None." % pdb_id)
        return None
    result = response.text
    info = json.loads(result)
    if store:
        with open(filename, 'wb') as file:
            pickle.dump(info, file)
        print('Finish downloading information for %s.' % pdb_id)
        return info
    else:
        return info


def get_ligands(pdbid, infoFile):
    """
    Get ligand names to a given PDB ID.
    """
    url_root = 'https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{}/{}'
    if not os.path.exists(infoFile):
        print('%s does not exist. Please fetch the structure information from RCSB.' % infoFile)
        return pdbid, None
    
    with open(infoFile, 'rb') as file:
        info = pickle.load(file)

    if info['rcsb_entry_info']['nonpolymer_entity_count'] == 0:
        # without binding small compounds
        return pdbid, []
    else:
        ligNames = []
        ligids = info['rcsb_entry_container_identifiers']['non_polymer_entity_ids']
        for lid in ligids:
            url = url_root.format(pdbid, lid)
            response = request_limited(url)
            if response is None or response.status_code != 200:
                print("%s-%s: Retrieval failed." % (pdbid, lid))
                continue
            text = json.loads(response.text)
            ligNames.append(text['pdbx_entity_nonpoly']['comp_id'])
        print('Finish fetching pdb-ligand information for %s.' % pdbid)
        return pdbid, ligNames


def download_ligand_file(ligname, dest, type='sdf', overwrite=True):
    """
    Download sdf file and cif file from RCSB PDB.
    """
    SDF_URL_BASE = 	'https://files.rcsb.org/ligands/view/{}_ideal.sdf'
    CIF_URL_BASE = 'https://files.rcsb.org/ligands/view/{}.cif'
    if os.path.exists(dest) and (not overwrite):
        print('%s already exists.' % dest)
        return True
    if type == 'sdf':
        url = SDF_URL_BASE.format(ligname)
    elif type == 'cif':
        url = CIF_URL_BASE.format(ligname)
    else:
        print(f"Invalid file type {type} for ligand {ligname}.")
        return False
    response = request_limited(url)
    if response is None or response.status_code != 200:
        print(f"Fail to fetch ligand {ligname}.{type}.")
        return False
    result = response.text
    with open(dest, 'w') as file:
        file.write(result)
    return True
