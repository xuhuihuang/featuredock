import argparse
import pymol

def align_and_save(mobilefile, reffile, outfile):
    pymol.cmd.load(mobilefile, 'mobile')
    pymol.cmd.load(reffile, 'ref')
    pymol.cmd.align('mobile', 'ref')
    pymol.cmd.save(outfile, 'mobile', -1, 'pdb')
    pymol.cmd.quit()


def align_and_save_with_ligand(mobilefile, mobligfile, reffile, outfile, outligfile):
    pymol.cmd.load(mobilefile, 'mobile')
    pymol.cmd.load(mobligfile, 'lig')
    # combine mobile protein and ligand
    pymol.cmd.copy_to('mobile', 'lig')
    # align together
    pymol.cmd.load(reffile, 'ref')
    pymol.cmd.align('mobile', 'ref')
    # save seperately: mobile, aligned_lig
    pymol.cmd.select('sele', 'resn UNK', domain='mobile')
    pymol.cmd.extract('aligned_lig', 'sele')
    pymol.cmd.save(outfile, 'mobile', -1, 'pdb')
    pymol.cmd.save(outligfile, 'aligned_lig', -1, 'sdf')
    pymol.cmd.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        This script aligns mobile pdb structure to a reference structure.
    """)
    sp = parser.add_subparsers(dest="sp")
    p1 = sp.add_parser('with_lig')
    p1.add_argument('--mobile', type=str, \
        help='path to the protein pdb file')
    p1.add_argument('--moblig', type=str, \
        help='path to the ligand sdf file')
    p1.add_argument('--ref', type=str, default='1b38.pdb', \
        help='path to the reference protein pdb file')
    p1.add_argument('--outfile', type=str, \
        help='path to store aligned mobile pdb file')
    p1.add_argument('--outligfile', type=str, \
        help='path to store aligned ligand sdf file')

    p2 = sp.add_parser('wo_lig')
    p2.add_argument('--mobile', type=str, \
        help='path to the pdb file')
    p2.add_argument('--ref', type=str, default='1b38.pdb', \
        help='path to the pdb file')
    p2.add_argument('--outfile', type=str, \
        help='path to store aligned mobile pdb file')
    args = parser.parse_args()

    if args.sp=='with_lig':
        align_and_save_with_ligand(args.mobile, args.moblig, \
            args.ref, args.outfile, args.outligfile)
    else:
        align_and_save(args.mobile, args.ref, args.outfile)
