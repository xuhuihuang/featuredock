import os
import subprocess
import argparse
import pymol
import shutil
pymol.finish_launching(['pymol', '-qc'])

def remove_hetatms(pid, pdbfile, outfile):
    pymol.cmd.load(pdbfile, pid)
    pymol.cmd.disable("all")
    pymol.cmd.enable(pid)
    pymol.cmd.select("hetatm")
    pymol.cmd.remove("sele")
    pymol.cmd.save(outfile, pid, -1, 'pdb')
    pymol.cmd.quit()
    

def remove_water(pid, pdbfile, outfile):
    pymol.cmd.load(pdbfile, pid)
    pymol.cmd.disable("all")
    pymol.cmd.enable(pid)
    pymol.cmd.select("resn HOH")
    pymol.cmd.remove("sele")
    pymol.cmd.save(outfile, pid, -1, 'pdb')
    pymol.cmd.quit()


def calc_dssp(apofile, dsspfile, dsspname):
    """
    Execute dssp in the command pipline.
    
    Compute the secondary structure for pdbfile, 
    and the output stores as dsspfile.
    dsspname: the name of program to compute dssp, either dssp or mkdssp
    """
    cmd = ("%s -i %s -o %s" % (dsspname, apofile, dsspfile))
    subprocess.Popen(cmd, shell=True).wait()


parser = argparse.ArgumentParser(description="remove water; calculate dssp")
parser.add_argument("--pdbid", type=str, help="PDBID")
parser.add_argument("--protfile", type=str, default=None, \
    help="path to protein file (including water mols) in PDBBind set")
parser.add_argument("--ligfile", type=str, default=None, \
    help="path to ligand file (including water mols) in PDBBind set")
parser.add_argument("--apodir", type=str, \
    help="Folder to contain apo protein file and dssp file")
parser.add_argument('--dssp', type=str, default='dssp', \
    help="DSSP program name or executive file path")
parser.add_argument("--hetdir", type=str, \
    help="Folder to contain heterogen sdf file")
parser.add_argument("--rm_all_het", action='store_true', \
    help="Remove all het atoms when flagged; otherwise remove water only")
args = parser.parse_args()

if not os.path.exists(args.apodir):
    os.makedirs(args.apodir, exist_ok=True)
    print(f"Folder created: {args.apodir}")

dsspname = args.dssp
pdbfile = args.protfile
sdffile = args.ligfile
apofile = os.path.join(args.apodir, f'{args.pdbid}.pdb')
dsspfile = os.path.join(args.apodir, f'{args.pdbid}.dssp')

if pdbfile:
    if args.rm_all_het:
        remove_hetatms(args.pdbid, pdbfile, apofile)
    else:
        remove_water(args.pdbid, pdbfile, apofile)

if args.dssp:
    calc_dssp(apofile, dsspfile, dsspname)

if sdffile:
    if not os.path.exists(args.hetdir):
        os.makedirs(args.hetdir, exist_ok=True)
        print(f"Folder created: {args.hetdir}")
    basefile = os.path.basename(sdffile)
    destfile = os.path.join(args.hetdir, basefile)
    shutil.copy(sdffile, destfile)
print(f"Finish processing {args.pdbid}")




