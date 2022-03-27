import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import gc
import common as cm
import re
import math
import time
import copy
import main
import pickle
import itertools as it
import traceback
from multiprocessing import Pool, Manager
import multiprocessing as mp
from scipy.ndimage.filters import gaussian_filter
from sklearn.cluster import KMeans


def valid(chunks):
    for chunk in chunks:
        print("1")
        mask = chunk['locus1_chrom'] == chunk['locus2_chrom']
        if mask.all():
            yield chunk
        else:
            yield chunk.loc[mask]
            break


def parse_hic(folder):
    if Path("pickle/hic_keys.gz").is_file():
        return joblib.load("pickle/hic_keys.gz")
    else:
        hic_keys = []
        directory = "hic"

        for filename in os.listdir(directory):
            fn = os.path.join(directory, filename)
            t_name = fn.replace("/", "_")
            print(t_name)
            if t_name not in ["hic_Ery.10kb.intra_chromosomal.interaction_table.tsv",
                              "hic_HUVEC.10kb.intra_chromosomal.interaction_table.tsv",
                              "hic_Islets.10kb.intra_chromosomal.interaction_table.tsv",
                              "hic_SkMC.10kb.intra_chromosomal.interaction_table.tsv"]:
                continue
            hic_keys.append(t_name)
            if Path(folder + t_name + "chr1").is_file():
                continue
            with open("hic.txt", "a+") as myfile:
                myfile.write(t_name)
            fields = ["locus1_chrom", "locus2_chrom", "locus1_start", "locus2_start", "pvalue"]
            dtypes = {"locus1_chrom": str, "locus2_chrom": str, "locus1_start": int, "locus2_start": int, "pvalue": str}
            chunksize = 10 ** 8
            chunks = pd.read_csv(fn, sep="\t", index_col=False, usecols=fields,
                                 dtype=dtypes, chunksize=chunksize, low_memory=True)
            df = pd.concat(valid(chunks))
            # df = pd.read_csv(fn, sep="\t", index_col=False, usecols=fields, dtype=dtypes, low_memory=True)
            df['pvalue'] = pd.to_numeric(df['pvalue'], errors='coerce')
            df['pvalue'].fillna(0, inplace=True)
            print(len(df))
            # No inter-chromosome connections are considered
            df.drop(df[df['locus1_chrom'] != df['locus2_chrom']].index, inplace=True)
            print(len(df))
            df.drop(['locus2_chrom'], axis=1, inplace=True)
            df.drop(df[df['locus1_start'] - df['locus2_start'] > 420000].index, inplace=True)
            print(len(df))

            # df["pvalue"] = -1 * np.log(df["pvalue"])
            # m = df.loc[df['pvalue'] != np.inf, 'pvalue'].max()
            # print("P Max is: " + str(m))
            # df['pvalue'].replace(np.inf, m, inplace=True)
            # df['pvalue'].clip(upper=100, inplace=True)
            # df["score"] = df["pvalue"] / df["pvalue"].max()
            df["score"] = df["logObservedOverExpected"] / df["logObservedOverExpected"].max()

            df.drop(["pvalue"], axis=1, inplace=True)
            chrd = list(df["locus1_chrom"].unique())
            for chr in chrd:
                joblib.dump(df.loc[df['locus1_chrom'] == chr].sort_values(by=['locus1_start']),
                            folder + t_name + chr, compress=3)
            print(t_name)
            with open("hic.txt", "a+") as myfile:
                myfile.write(t_name)
            del df
            gc.collect()

        joblib.dump(hic_keys, "pickle/hic_keys.gz", compress=3)
        chromosomes = ["chrX", "chrY"]
        for i in range(1, 23):
            chromosomes.append("chr" + str(i))
        for key in hic_keys:
            print(key)
            hdf = {}
            for chr in chromosomes:
                try:
                    hdf[chr] = joblib.load(folder + key + chr)
                except:
                    pass
            joblib.dump(hdf, folder + key, compress=3)
            print(key)
        return hic_keys


def parse_tracks(ga, bin_size, tss_loc, chromosomes, tracks_folder):
    track_names = []
    for filename in os.listdir(tracks_folder):
        if filename.endswith(".gz"):
            track = filename[:-len(".100nt.bed.gz")]
            fn = tracks_folder + f"{track}.100nt.bed.gz"
            size = os.path.getsize(fn)
            if size > 2 * 512000 or track.startswith("sc"):
                track_names.append(track)

    print(f"gas {len(track_names)}")

    step_size = 50
    q = mp.Queue()
    ps = []
    start = 0
    nproc = 28
    end = len(track_names)
    for t in range(start, end, step_size):
        t_end = min(t+step_size, end)
        sub_tracks = track_names[t:t_end]
        p = mp.Process(target=parse_some_tracks,
                       args=(q, sub_tracks, ga, bin_size, chromosomes,tracks_folder,))
        p.start()
        ps.append(p)
        if len(ps) >= nproc:
            for p in ps:
                p.join()
            print(q.get())
            ps = []

    if len(ps) > 0:
        for p in ps:
            p.join()
        print(q.get())

    joblib.dump(track_names, "pickle/track_names.gz", compress=3)
    return track_names


def parse_some_tracks(q, some_tracks, ga, bin_size, chromosomes, tracks_folder):
    for track in some_tracks:
        try:
            fn = tracks_folder + f"{track}.100nt.bed.gz"
            gast = copy.deepcopy(ga)
            dtypes = {"chr": str, "start": int, "end": int, "score": float}
            df = pd.read_csv(fn, delim_whitespace=True, names=["chr", "start", "end", "score"],
                             dtype=dtypes, header=None, index_col=False)

            chrd = list(df["chr"].unique())
            df["mid"] = (df["start"] + (df["end"] - df["start"]) / 2) / bin_size
            df = df.astype({"mid": int})

            # group the scores over `key` and gather them in a list
            grouped_scores = df.groupby("chr").agg(list)

            # for each key, value in the dictionary...
            for key, val in gast.items():
                if key not in chrd:
                    continue
                # first lookup the positions to update and the corresponding scores
                pos, score = grouped_scores.loc[key, ["mid", "score"]]
                # fancy indexing
                gast[key][pos] += score

            max_val = -1
            # all_vals = None
            for key in gast.keys():
                if "scEnd5" in track:
                    gast[key] = np.log10(np.exp(gast[key]))
                else:
                    gast[key] = np.log10(gast[key] + 1)

                if key in chromosomes:
                    max_val = max(np.max(gast[key]), max_val)
                    # if all_vals is not None:
                    #     all_vals = np.concatenate((all_vals, gast[key][tss_loc[key]]))
                    # else:
                    #     all_vals = gast[key][tss_loc[key]]
            # tss_loc_num = len(all_vals)
            # all_vals = all_vals[all_vals != 0]
            # all_vals.sort()
            # scale_val = all_vals[int(0.95 * len(all_vals))]
            # if scale_val == 0:
            #     print(scale_val)
            for key in gast.keys():
                gast[key] = gast[key] / max_val  # np.clip(gast[key], 0, scale_val) / scale_val
                gast[key] = gaussian_filter(gast[key], sigma=0.5)
            joblib.dump(gast, main.p.parsed_tracks_folder + track, compress="lz4")
            # pickle.dump(gast, open(main.p.parsed_tracks_folder + track, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Parsed {track}. Max value: {max_val}.")
        except Exception as exc:
            print(exc)
            traceback.print_exc()
            print("\n\n\nCould not parse! " + track)
    q.put(None)


def get_sequences(bin_size, chromosomes):
    if Path("pickle/genome.gz").is_file():
        genome = joblib.load("pickle/genome.gz")
        ga = joblib.load("pickle/ga.gz")
    else:
        genome, ga = cm.parse_genome("data/hg38.fa", bin_size)
        joblib.dump(genome, "pickle/genome.gz", compress=3)
        joblib.dump(ga, "pickle/ga.gz", compress=3)

    if Path("pickle/train_info.gz").is_file():
        test_info = joblib.load("pickle/test_info.gz")
        train_info = joblib.load("pickle/train_info.gz")
        tss_loc = joblib.load("pickle/tss_loc.gz")
        protein_coding = joblib.load("pickle/protein_coding.gz")
    else:
        # gene_tss = pd.read_csv("data/old_TSS_flank_0.bed",
        #                     sep="\t", index_col=False, names=["chrom", "start", "end", "geneID", "score", "strand"])
        # gene_info = pd.read_csv("data/old_gene.info.tsv", sep="\t", index_col=False)

        gene_tss = pd.read_csv("data/hg38.GENCODEv38.pc_lnc.TSS.bed", sep="\t", index_col=False,
                               names=["chrom", "start", "end", "geneID", "score", "strand"])
        gene_info = pd.read_csv("data/hg38.GENCODEv38.pc_lnc.gene.info.tsv", sep="\t", index_col=False)
        protein_coding = []
        # prom_info = pd.read_csv("data/hg38.gencode_v32.promoter.window.info.tsv", sep="\t", index_col=False)
        test_info = []
        tss_loc = {}
        # test_genes = prom_info.loc[(prom_info['chrom'] == "chr1") & (prom_info['max_overall_rank'] == 1)]
        # for index, row in test_genes.iterrows():
        #     vals = row["TSS_str"].split(";")
        #     pos = int(vals[int(len(vals) / 2)].split(",")[1])
        #     strand = vals[int(len(vals) / 2)].split(",")[2]
        #     test_info.append([row["chrom"], pos, row["geneID_str"], row["geneType_str"], strand])
        test_genes = gene_tss.loc[gene_tss['chrom'] == "chr1"]
        for index, row in test_genes.iterrows():
            pos = int(row["start"])
            gene_type = gene_info[gene_info['geneID'] == row["geneID"]]['geneType'].values[0]
            gene_name = gene_info[gene_info['geneID'] == row["geneID"]]['geneName'].values[0]
            if gene_type == "protein_coding":
                protein_coding.append(row["geneID"])
            # if gene_type != "protein_coding":
            #     continue
            tss_loc.setdefault(row["chrom"], []).append(pos)
            test_info.append([row["chrom"], pos, row["geneID"], gene_type,
                              row["strand"], gene_type != "protein_coding", gene_name])

        print(f"Test set complete {len(test_info)}")
        train_info = []
        train_genes = gene_tss.loc[gene_tss['chrom'] != "chr1"]
        for index, row in train_genes.iterrows():
            pos = int(row["start"])
            gene_type = gene_info[gene_info['geneID'] == row["geneID"]]['geneType'].values[0]
            gene_name = gene_info[gene_info['geneID'] == row["geneID"]]['geneName'].values[0]
            if gene_type == "protein_coding":
                protein_coding.append(row["geneID"])
            # if gene_type != "protein_coding":
            #     continue
            tss_loc.setdefault(row["chrom"], []).append(pos)
            if row["chrom"] not in chromosomes:
                continue
            train_info.append([row["chrom"], pos, row["geneID"], gene_type, row["strand"],
                               gene_type != "protein_coding", gene_name])

        print(f"Training set complete {len(train_info)}")

        one_hot = {}
        for chromosome in chromosomes:
            print(chromosome)
            one_hot[chromosome] = cm.encode_seq(genome[chromosome])
            ######################################################################
            tss_layer = np.zeros((len(one_hot[chromosome]), 1)).astype(bool)  #
            print(len(one_hot[chromosome]))  #
            for tss in tss_loc[chromosome]:  #
                tss_layer[tss, 0] = True  #
            print(f"{chromosome}: {np.sum(tss_layer)}")  #
            one_hot[chromosome] = np.hstack([one_hot[chromosome], tss_layer])  #
            ######################################################################

        joblib.dump(one_hot, "pickle/one_hot.gz", compress=3)

        joblib.dump(test_info, "pickle/test_info.gz", compress=3)
        joblib.dump(train_info, "pickle/train_info.gz", compress=3)
        joblib.dump(tss_loc, "pickle/tss_loc.gz", compress=3)
        joblib.dump(protein_coding, "pickle/protein_coding.gz", compress=3)
        gc.collect()
    one_hot = joblib.load("pickle/one_hot.gz")
    return ga, one_hot, train_info, test_info, tss_loc, protein_coding


def parse_eval_data(chromosomes):
    if not Path("pickle/train_info_eval.gz").is_file():
        # gene_tss = pd.read_csv("data/old_TSS_flank_0.bed",
        #                     sep="\t", index_col=False, names=["chrom", "start", "end", "geneID", "score", "strand"])
        # gene_info = pd.read_csv("data/old_gene.info.tsv", sep="\t", index_col=False)
        gene_tss = pd.read_csv("data/hg38.GENCODEv38.pc_lnc.TSS.bed", sep="\t", index_col=False,
                               names=["chrom", "start", "end", "geneID", "score", "strand"])
        gene_info = pd.read_csv("data/hg38.GENCODEv38.pc_lnc.gene.info.tsv", sep="\t", index_col=False)
        test_info = []
        test_genes = gene_tss.loc[gene_tss['chrom'] == "chr1"]
        for index, row in test_genes.iterrows():
            pos = int(row["start"]) - 1
            gene_type = gene_info[gene_info['geneID'] == row["geneID"]]['geneType'].values[0]
            gene_name = gene_info[gene_info['geneID'] == row["geneID"]]['geneName'].values[0]
            test_info.append([row["chrom"], pos, row["geneID"], gene_type,
                              row["strand"], gene_type != "protein_coding", gene_name])

        print(f"Test set complete {len(test_info)}")
        train_info = []
        train_genes = gene_tss.loc[gene_tss['chrom'] != "chr1"]
        for index, row in train_genes.iterrows():
            pos = int(row["start"]) - 1
            gene_type = gene_info[gene_info['geneID'] == row["geneID"]]['geneType'].values[0]
            gene_name = gene_info[gene_info['geneID'] == row["geneID"]]['geneName'].values[0]
            if row["chrom"] not in chromosomes:
                continue
            train_info.append([row["chrom"], pos, row["geneID"], gene_type, row["strand"],
                               gene_type != "protein_coding", gene_name])

        print(f"Training set complete {len(train_info)}")

        joblib.dump(test_info, "pickle/test_info_eval.gz", compress=3)
        joblib.dump(train_info, "pickle/train_info_eval.gz", compress=3)


def parse_one_track(ga, bin_size, fn):
    gast = copy.deepcopy(ga)
    dtypes = {"chr": str, "start": int, "end": int, "score": float}
    df = pd.read_csv(fn, delim_whitespace=True, names=["chr", "start", "end", "score"],
                     dtype=dtypes, header=None, index_col=False)

    chrd = list(df["chr"].unique())
    df["mid"] = (df["start"] + (df["end"] - df["start"]) / 2) / bin_size
    df = df.astype({"mid": int})

    # group the scores over `key` and gather them in a list
    grouped_scores = df.groupby("chr").agg(list)

    # for each key, value in the dictionary...
    for key, val in gast.items():
        if key not in chrd:
            continue
        # first lookup the positions to update and the corresponding scores
        pos, score = grouped_scores.loc[key, ["mid", "score"]]
        # fancy indexing
        gast[key][pos] += score

    max_val = -1
    for key in gast.keys():
        gast[key] = np.log(gast[key] + 1)
        max_val = max(np.max(gast[key]), max_val)
    for key in gast.keys():
        gast[key] = gast[key] / max_val

    return gast
