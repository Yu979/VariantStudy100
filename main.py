import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,2'
import math
import logging
import joblib
import gc
import random
import time
import pandas as pd
# pip install modin[all]
# import modin.pandas as pd
import numpy as np
import common as cm
from pathlib import Path
import matplotlib
import psutil
import sys
import parse_data as parser
from datetime import datetime
import traceback
import multiprocessing as mp
import evaluation
from main_params import MainParams
from scipy.ndimage import gaussian_filter
import cooler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
matplotlib.use("agg")


def load_old_weights():
    import model as mo
    our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, hic_num, heads["hg38"])
    our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
    our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression_hg38"))
    our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_path + "_epigenome"))
    our_model.get_layer("our_conservation").set_weights(joblib.load(p.model_path + "_conservation"))
    our_model.get_layer("our_hic").set_weights(joblib.load(p.model_path + "_hic"))

    our_model_old = mo.make_model(5120, p.num_features, 20, 0, heads["hg38"])
    our_model_old.get_layer("our_resnet").set_weights(joblib.load(p.model_folder + "our_model_5120" + "_res"))
    our_model.get_layer("our_expression").set_weights(
        joblib.load(p.model_folder + "our_model_5120" + "_expression_hg38"))
    our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_folder + "our_model_5120" + "_epigenome"))
    our_model.get_layer("our_conservation").set_weights(
        joblib.load(p.model_folder + "our_model_5120" + "_conservation"))

    print("model loaded")
    for layer in our_model_old.get_layer("our_resnet").layers:
        layer_name = layer.name
        layer_weights = layer.weights
        try:
            our_model.get_layer("our_resnet").get_layer(layer_name).set_weights(layer_weights)
        except:
            print(layer_name)

    for head_type in heads["hg38"].keys():
        head_type = "our_" + head_type
        for layer in our_model_old.get_layer(head_type).layers:
            layer_name = layer.name
            layer_weights = layer.weights
            try:
                our_model.get_layer(head_type).get_layer(layer_name).set_weights(layer_weights)
            except:
                print(layer_name)

    safe_save(our_model.get_layer("our_resnet").get_weights(), p.model_path + "_res")
    safe_save(our_model.get_layer("our_expression").get_weights(), p.model_path + "_expression_hg38")
    safe_save(our_model.get_layer("our_hic").get_weights(), p.model_path + "_hic")
    safe_save(our_model.get_layer("our_epigenome").get_weights(), p.model_path + "_epigenome")
    safe_save(our_model.get_layer("our_conservation").get_weights(), p.model_path + "_conservation")
    print("new weights saved")


def get_data_and_train(last_proc, fit_epochs, head_id):
    print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(current_epoch) + " " + p.species[head_id])
    training_regions = joblib.load(f"{p.pickle_folder}{p.species[head_id]}_regions.gz")
    one_hot = joblib.load(f"{p.pickle_folder}{p.species[head_id]}_one_hot.gz")
    # training regions are shuffled each iteration
    shuffled_regions_info = random.sample(training_regions, len(training_regions))
    input_sequences = []
    output_scores_info = []
    picked_regions = []
    for i, info in enumerate(shuffled_regions_info):
        if len(input_sequences) >= p.GLOBAL_BATCH_SIZE * p.STEPS_PER_EPOCH:
            break
        # Don't use chrY, chrM etc
        if info[0] not in one_hot.keys():
            continue
        start = info[1] - (info[1] % p.bin_size) - p.half_size
        extra = start + p.input_size - len(one_hot[info[0]])
        if start < 0 or extra > 0:
            continue
        ns = one_hot[info[0]][start:start + p.input_size]
        if np.any(ns[:, -1]):
            # Test tss or exclude region was encountered! Skipping
            continue

        dry_run_regions.setdefault(p.species[head_id], []).append(
            f"{info[0]}\t{start}\t{start + p.input_size}\ttrain")
        picked_regions.append(info)
        start_bin = int(info[1] / p.bin_size) - p.half_num_regions
        input_sequences.append(ns)
        output_scores_info.append([info[0], start_bin, start_bin + p.num_bins])
    print(datetime.now().strftime('[%H:%M:%S] ') + "Loading parsed tracks")
    if head_id == 0:
        output_conservation = parser.par_load_data(output_scores_info, heads[p.species[head_id]]["conservation"], p)
        gc.collect()
        output_expression = parser.par_load_data(output_scores_info, heads[p.species[head_id]]["expression"], p)
        gc.collect()
        # a = np.max(output_expression, axis=-2)
        # a = np.sum(output_expression, axis=-1)
        # a = np.min(a, axis=0)
        output_epigenome = parser.par_load_data(output_scores_info, heads[p.species[head_id]]["epigenome"], p)
        gc.collect()
    else:
        output_expression = parser.par_load_data(output_scores_info, heads[p.species[head_id]], p)
    gc.collect()
    print("")
    # half of sequences will be reverse-complemented
    half = len(input_sequences) // 2
    # loading corresponding 2D data
    output_hic = parser.par_load_hic_data(hic_keys, p, picked_regions, half)
    gc.collect()
    print_memory()
    input_sequences = np.asarray(input_sequences, dtype=bool)

    # reverse-complement
    with mp.Pool(8) as pool:
        rc_arr = pool.map(change_seq, input_sequences[:half])
    input_sequences[:half] = rc_arr

    # for reverse-complement sequences, the 1D output is flipped
    for i in range(half):
        output_expression[i] = np.flip(output_expression[i], axis=1)
        if p.species[head_id] == "hg38":
            output_epigenome[i] = np.flip(output_epigenome[i], axis=1)
            output_conservation[i] = np.flip(output_conservation[i], axis=1)

    all_outputs = {"our_expression": output_expression}
    if p.species[head_id] == "hg38":
        all_outputs["our_epigenome"] = output_epigenome
        all_outputs["our_conservation"] = output_conservation
        if hic_num > 0:
            all_outputs["our_hic"] = output_hic
    # Cut off the test TSS layer
    input_sequences = input_sequences[:, :, :-1]
    print(input_sequences.shape)

    gc.collect()
    print_memory()
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                             key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, cm.get_human_readable(size)))

    print(datetime.now().strftime('[%H:%M:%S] ') + "Training")

    if last_proc is not None:
        print("Waiting")
        print(mp_q.get())
        last_proc.join()
        print("Finished waiting")
    # joblib.dump([input_sequences, all_outputs], f"{p.pickle_folder}run.gz", compress=3)
    # argss = joblib.load(f"{p.pickle_folder}run.gz")
    # p = mp.Process(target=train_step, args=(argss[0][:400], argss[1][:400], argss[2][:400], fit_epochs, head_id,))
    proc = mp.Process(target=make_model_and_train, args=(
        heads[p.species[head_id]], p.species[head_id], input_sequences, all_outputs, fit_epochs, hic_num, mp_q,))
    proc.start()
    return proc


def safe_save(thing, place):
    joblib.dump(thing, place + "_temp", compress="lz4")
    if os.path.exists(place):
        os.remove(place)
    os.rename(place + "_temp", place)


def make_model_and_train(head, head_name, input_sequences, all_outputs, fit_epochs, hic_num, mp_q):
    print(f"=== Training with head {head_name} ===")
    import tensorflow as tf
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    import tensorflow_addons as tfa
    import model as mo
    try:
        if head_name == "hg38":
            train_data = mo.wrap_for_human_training(input_sequences, all_outputs, p.GLOBAL_BATCH_SIZE)
        else:
            train_data = mo.wrap(input_sequences, all_outputs["our_expression"], p.GLOBAL_BATCH_SIZE)
        del input_sequences
        del all_outputs
        gc.collect()
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        with strategy.scope():
            our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, hic_num if head_name == "hg38" else 0,
                                      p.hic_size, head)
            # if current_epoch == 0:
            #     our_model.get_layer("our_resnet").trainable = False
            #     our_model.get_layer("our_expression").trainable = False
            #     our_model.get_layer("our_epigenome").trainable = False
            #     our_model.get_layer("our_conservation").trainable = False
            #     our_model.get_layer("our_hic").trainable = True
            #     print(our_model.summary())

            # preparing the main optimizer
            # tf.keras.optimizers.Adam(learning_rate=learning_rates["our_expression"], clipnorm=0.001)
            # "our_resnet": tfa.optimizers.AdamW(learning_rate=learning_rates["our_resnet"], weight_decay=0.00001, clipnorm=0.001)

            optimizers_and_layers = [(optimizers["our_resnet"], our_model.get_layer("our_resnet")),
                                     (optimizers["our_expression"], our_model.get_layer("our_expression"))]
            if head_name == "hg38":
                if hic_num > 0:
                    optimizers_and_layers.append((optimizers["our_hic"], our_model.get_layer("our_hic")))
                optimizers_and_layers.append((optimizers["our_epigenome"], our_model.get_layer("our_epigenome")))
                optimizers_and_layers.append((optimizers["our_conservation"], our_model.get_layer("our_conservation")))
            optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
            # loading the loss weights and compiling the model
            if head_name == "hg38":
                losses = {
                    "our_expression": mo.fast_mse,
                    "our_epigenome": "mse",
                    "our_conservation": "mse",
                }
                if hic_num > 0:
                    losses["our_hic"] = "mse"
                our_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)
            else:
                our_model.compile(optimizer=optimizer, loss="mse")
            if current_epoch == 0 and os.path.exists(p.model_path + "_opt_resnet"):
                # need to init the optimizers
                our_model.fit(train_data, steps_per_epoch=1, epochs=1)
                # loading the previous optimizer weights
                if os.path.exists(p.model_path + "_opt_expression_" + head_name):
                    print("loading expression optimizer")
                    optimizers["our_expression"].set_weights(joblib.load(p.model_path + "_opt_expression_" + head_name))
                    optimizers["our_expression"].learning_rate.assign(learning_rates["our_expression"])
                if os.path.exists(p.model_path + "_opt_resnet"):
                    print("loading resnet optimizer")
                    optimizers["our_resnet"].set_weights(joblib.load(p.model_path + "_opt_resnet"))
                    optimizers["our_resnet"].learning_rate.assign(learning_rates["our_resnet"])
                if head_name == "hg38":
                    if hic_num > 0 and os.path.exists(p.model_path + "_opt_hic"):
                        print("loading hic optimizer")
                        optimizers["our_hic"].set_weights(joblib.load(p.model_path + "_opt_hic"))
                        optimizers["our_hic"].learning_rate.assign(learning_rates["our_hic"])

                    if os.path.exists(p.model_path + "_opt_epigenome"):
                        print("loading epigenome optimizer")
                        optimizers["our_epigenome"].set_weights(joblib.load(p.model_path + "_opt_epigenome"))
                        optimizers["our_epigenome"].learning_rate.assign(learning_rates["our_epigenome"])

                    if os.path.exists(p.model_path + "_opt_conservation"):
                        print("loading conservation optimizer")
                        optimizers["our_conservation"].set_weights(joblib.load(p.model_path + "_opt_conservation"))
                        optimizers["our_conservation"].learning_rate.assign(learning_rates["our_conservation"])
            # loading model weights
            if os.path.exists(p.model_path + "_res"):
                our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
            if os.path.exists(p.model_path + "_expression_" + head_name):
                our_model.get_layer("our_expression").set_weights(
                    joblib.load(p.model_path + "_expression_" + head_name))
            if head_name == "hg38" and os.path.exists(p.model_path + "_epigenome"):
                our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_path + "_epigenome"))
            if head_name == "hg38" and hic_num > 0 and os.path.exists(p.model_path + "_hic"):
                our_model.get_layer("our_hic").set_weights(joblib.load(p.model_path + "_hic"))
            if head_name == "hg38" and os.path.exists(p.model_path + "_conservation"):
                our_model.get_layer("our_conservation").set_weights(
                    joblib.load(p.model_path + "_conservation"))
    except:
        traceback.print_exc()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Error while compiling.")
        mp_q.put(None)
        return None

    try:
        our_model.fit(train_data, epochs=fit_epochs, batch_size=p.GLOBAL_BATCH_SIZE)
        del train_data
        gc.collect()
        safe_save(our_model.get_layer("our_resnet").get_weights(), p.model_path + "_res")
        safe_save(optimizers["our_resnet"].get_weights(), p.model_path + "_opt_resnet")
        safe_save(our_model.get_layer("our_expression").get_weights(), p.model_path + "_expression_" + head_name)
        safe_save(optimizers["our_expression"].get_weights(), p.model_path + "_opt_expression_" + head_name)
        if head_name == "hg38":
            if hic_num > 0:
                safe_save(optimizers["our_hic"].get_weights(), p.model_path + "_opt_hic")
                safe_save(our_model.get_layer("our_hic").get_weights(), p.model_path + "_hic")
            safe_save(optimizers["our_epigenome"].get_weights(), p.model_path + "_opt_epigenome")
            safe_save(our_model.get_layer("our_epigenome").get_weights(), p.model_path + "_epigenome")
            safe_save(optimizers["our_conservation"].get_weights(), p.model_path + "_opt_conservation")
            safe_save(our_model.get_layer("our_conservation").get_weights(), p.model_path + "_conservation")
    except:
        traceback.print_exc()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Error while training.")
        mp_q.put(None)
        return None

    print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch finished. ")
    mp_q.put(None)
    return None


def check_perf(mp_q):
    print(datetime.now().strftime('[%H:%M:%S] ') + "Evaluating")
    import tensorflow as tf
    import model as mo
    one_hot = joblib.load(f"{p.pickle_folder}hg38_one_hot.gz")
    try:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        with strategy.scope():
            our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, heads["hg38"])
            our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
            our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression_hg38"))
            our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_path + "_epigenome"))
            our_model.get_layer("our_conservation").set_weights(joblib.load(p.model_path + "_conservation"))
        train_eval_chr = "chr1"
        train_eval_chr_info = []
        for info in train_info:
            if info[0] == train_eval_chr:  # and 139615843 < info[1] < 141668489
                train_eval_chr_info.append(info)
        print(f"Training set {len(train_eval_chr_info)}")
        training_spearman = evaluation.eval_perf(p, our_model, heads["hg38"], train_eval_chr_info,
                                                 False, current_epoch, "train", one_hot)
        # training_spearman = 0
        test_eval_chr_info = []
        for info in test_info:
            # if info[0] == "chr11":
            test_eval_chr_info.append(info)
        print(f"Test set {len(test_eval_chr_info)}")
        test_spearman = evaluation.eval_perf(p, our_model, heads["hg38"], test_eval_chr_info,
                                             False, current_epoch, "test", one_hot)
        with open(p.model_name + "_history.csv", "a+") as myfile:
            myfile.write(f"{training_spearman},{test_spearman}")
            myfile.write("\n")
    except Exception as e:
        print(e)
        traceback.print_exc()
        print(datetime.now().strftime('[%H:%M:%S] ') + "Problem during evaluation")
    mp_q.put(None)


def print_memory():
    mem = psutil.virtual_memory()
    print(f"used: {cm.get_human_readable(mem.used)} available: {cm.get_human_readable(mem.available)}")


def change_seq(x):
    return cm.rev_comp(x)


last_proc = None
p = MainParams()
dry_run_regions = {}
if __name__ == '__main__':
    # import ray
    # ray.init()
    train_info, test_info, protein_coding = parser.parse_sequences(p)
    if Path(f"{p.pickle_folder}track_names_col.gz").is_file():
        track_names_col = joblib.load(f"{p.pickle_folder}track_names_col.gz")
    else:
        track_names_col = parser.parse_tracks(p)

    if Path(f"{p.pickle_folder}heads.gz").is_file():
        heads = joblib.load(f"{p.pickle_folder}heads.gz")
    else:
        heads = {}
        for specie in p.species:
            shuffled_tracks = random.sample(track_names_col[specie], len(track_names_col[specie]))
            if specie == "hg38":
                meta = pd.read_csv("/data2/qinze/geneExpression/media/user/PASSPORT1/variants_100/data/ML_all_track.metadata.2022053017.tsv", sep="\t")
                new_head = {"expression": [x for x in shuffled_tracks if
                                           meta.loc[meta['file_name'] == x].iloc[0]["technology"].startswith(
                                               ("CAGE", "RAMPAGE", "NETCAGE"))],
                            "epigenome": [x for x in shuffled_tracks if
                                          meta.loc[meta['file_name'] == x].iloc[0]["technology"].startswith(
                                              ("DNase", "ATAC", "Histone_ChIP", "TF_ChIP"))],
                            "conservation": [x for x in shuffled_tracks if
                                             meta.loc[meta['file_name'] == x].iloc[0]["value"].startswith(
                                                 "conservation")]}
            else:
                new_head = shuffled_tracks
            heads[specie] = new_head
        joblib.dump(heads, f"{p.pickle_folder}heads.gz", compress="lz4")

    for head_key in heads.keys():
        if head_key == "hg38":
            for human_key in heads[head_key]:
                print(f"Number of tracks in head {head_key} {human_key}: {len(heads[head_key][human_key])}")
        else:
            print(f"Number of tracks in head {head_key}: {len(heads[head_key])}")

    hic_keys = pd.read_csv("/data2/qinze/geneExpression/media/user/PASSPORT1/variants_100/data/good_hic.tsv", sep="\t", header=None).iloc[:, 0]
    # hic_keys = []
    hic_num = len(hic_keys)
    print(f"hic {hic_num}")

    # import model as mo
    # our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, hic_num, p.hic_size, heads["hg38"])
    import tensorflow_addons as tfa

    loss_weights = {}
    learning_rates = {}
    weight_decays = {}
    with open(str(p.script_folder) + "/loss_weights") as f:
        for line in f:
            (key, weight, lr, wd) = line.split()
            if hic_num == 0 and key == "our_hic":
                continue
            if key != "our_resnet":
                loss_weights[key] = float(weight)
            learning_rates[key] = float(lr)
            weight_decays[key] = float(wd)
    optimizers = {}
    optimizers["our_resnet"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_resnet"],
                                                    weight_decay=weight_decays["our_resnet"], clipnorm=0.001)
    optimizers["our_expression"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_expression"],
                                                        weight_decay=weight_decays["our_expression"], clipnorm=0.001)
    optimizers["our_epigenome"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_epigenome"],
                                                       weight_decay=weight_decays["our_epigenome"], clipnorm=0.001)
    optimizers["our_conservation"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_conservation"],
                                                          weight_decay=weight_decays["our_conservation"],
                                                          clipnorm=0.001)
    optimizers["our_hic"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_hic"],
                                                 weight_decay=weight_decays["our_hic"], clipnorm=0.001)

    # load_old_weights()
    # exit()
    mp_q = mp.Queue()
    print("Training starting")
    start_epoch = 0
    fit_epochs = 4
    try:
        for current_epoch in range(start_epoch, p.num_epochs, 1):
            head_id = 0
            # with open(str(p.script_folder) + "/../step_num") as f:
            #     p.STEPS_PER_EPOCH = int(f.read())
            # if current_epoch % 2 == 0:
            #     head_id = 0
            # else:
            #     head_id = 1 + (current_epoch - math.ceil(current_epoch / 2)) % (len(heads) - 1)
            # if head_id == 0:
            #     p.STEPS_PER_EPOCH = 400
            #     fit_epochs = 4
            # else:
            #     p.STEPS_PER_EPOCH = 600
            #     fit_epochs = 2

            # check_perf(mp_q)
            # exit()
            last_proc = get_data_and_train(last_proc, fit_epochs, head_id)
            if current_epoch % 20 == 0:  # and current_epoch != 0:
                print("Eval epoch")
                print(mp_q.get())
                last_proc.join()
                last_proc = None
                proc = mp.Process(target=check_perf, args=(mp_q,))
                proc.start()
                print(mp_q.get())
                proc.join()
    except Exception as e:
        traceback.print_exc()

# for key in dry_run_regions.keys():
#     with open(f"dry/{key}_dry_test.bed", "w") as text_file:
#         text_file.write("\n".join(dry_run_regions[key]))

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,2'
# import math
# import logging
# import joblib
# import gc
# import random
# import time
# import pandas as pd
# # pip install modin[all]
# # import modin.pandas as pd
# import numpy as np
# import common as cm
# from pathlib import Path
# import matplotlib
# import psutil
# import sys
# import parse_data as parser
# from datetime import datetime
# import traceback
# import multiprocessing as mp
# import evaluation
# from main_params import MainParams
# from scipy.ndimage import gaussian_filter
# import cooler
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
# matplotlib.use("agg")
#
#
# def load_old_weights():
#     import model as mo
#     our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, hic_num, heads["hg38"])
#     our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
#     our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression_hg38"))
#     our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_path + "_epigenome"))
#     our_model.get_layer("our_conservation").set_weights(joblib.load(p.model_path + "_conservation"))
#     our_model.get_layer("our_hic").set_weights(joblib.load(p.model_path + "_hic"))
#
#     our_model_old = mo.make_model(5120, p.num_features, 20, 0, heads["hg38"])
#     our_model_old.get_layer("our_resnet").set_weights(joblib.load(p.model_folder + "our_model_5120" + "_res"))
#     our_model.get_layer("our_expression").set_weights(joblib.load(p.model_folder + "our_model_5120" + "_expression_hg38"))
#     our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_folder + "our_model_5120" + "_epigenome"))
#     our_model.get_layer("our_conservation").set_weights(joblib.load(p.model_folder + "our_model_5120" + "_conservation"))
#
#     print("model loaded")
#     for layer in our_model_old.get_layer("our_resnet").layers:
#         layer_name = layer.name
#         layer_weights = layer.weights
#         try:
#             our_model.get_layer("our_resnet").get_layer(layer_name).set_weights(layer_weights)
#         except:
#             print(layer_name)
#
#     for head_type in heads["hg38"].keys():
#         head_type = "our_" + head_type
#         for layer in our_model_old.get_layer(head_type).layers:
#             layer_name = layer.name
#             layer_weights = layer.weights
#             try:
#                 our_model.get_layer(head_type).get_layer(layer_name).set_weights(layer_weights)
#             except:
#                 print(layer_name)
#
#     safe_save(our_model.get_layer("our_resnet").get_weights(), p.model_path + "_res")
#     safe_save(our_model.get_layer("our_expression").get_weights(), p.model_path + "_expression_hg38")
#     safe_save(our_model.get_layer("our_hic").get_weights(), p.model_path + "_hic")
#     safe_save(our_model.get_layer("our_epigenome").get_weights(), p.model_path + "_epigenome")
#     safe_save(our_model.get_layer("our_conservation").get_weights(), p.model_path + "_conservation")
#     print("new weights saved")
#
#
# def get_data_and_train(last_proc, fit_epochs, head_id):
#     print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch " + str(current_epoch) + " " + p.species[head_id])
#     training_regions = joblib.load(f"{p.pickle_folder}{p.species[head_id]}_regions.gz")
#     one_hot = joblib.load(f"{p.pickle_folder}{p.species[head_id]}_one_hot.gz")
#     # training regions are shuffled each iteration
#     shuffled_regions_info = random.sample(training_regions, len(training_regions))
#     input_sequences = []
#     output_scores_info = []
#     picked_regions = []
#     for i, info in enumerate(shuffled_regions_info):
#         if len(input_sequences) >= p.GLOBAL_BATCH_SIZE * p.STEPS_PER_EPOCH:
#             break
#         # Don't use chrY, chrM etc
#         if info[0] not in one_hot.keys():
#             continue
#         start = info[1] - (info[1] % p.bin_size) - p.half_size
#         extra = start + p.input_size - len(one_hot[info[0]])
#         if start < 0 or extra > 0:
#             continue
#         ns = one_hot[info[0]][start:start + p.input_size]
#         if np.any(ns[:, -1]):
#             # Test tss or exclude region was encountered! Skipping
#             continue
#
#         dry_run_regions.setdefault(p.species[head_id], []).append(
#             f"{info[0]}\t{start}\t{start + p.input_size}\ttrain")
#         picked_regions.append(info)
#         start_bin = int(info[1] / p.bin_size) - p.half_num_regions
#         input_sequences.append(ns)
#         output_scores_info.append([info[0], start_bin, start_bin + p.num_bins])
#     print(datetime.now().strftime('[%H:%M:%S] ') + "Loading parsed tracks")
#     if head_id == 0:
#         output_conservation = parser.par_load_data(output_scores_info, heads[p.species[head_id]]["conservation"], p)
#         gc.collect()
#         output_expression = parser.par_load_data(output_scores_info, heads[p.species[head_id]]["expression"], p)
#         gc.collect()
#         # a = np.max(output_expression, axis=-2)
#         # a = np.sum(output_expression, axis=-1)
#         # a = np.min(a, axis=0)
#         output_epigenome = parser.par_load_data(output_scores_info, heads[p.species[head_id]]["epigenome"], p)
#         gc.collect()
#     else:
#         output_expression = parser.par_load_data(output_scores_info, heads[p.species[head_id]], p)
#     gc.collect()
#     print("")
#     # half of sequences will be reverse-complemented
#     half = len(input_sequences) // 2
#     # loading corresponding 2D data
#     output_hic = parser.par_load_hic_data(hic_keys, p, picked_regions, half)
#     gc.collect()
#     print_memory()
#     input_sequences = np.asarray(input_sequences, dtype=bool)
#
#     # reverse-complement
#     with mp.Pool(8) as pool:
#         rc_arr = pool.map(change_seq, input_sequences[:half])
#     input_sequences[:half] = rc_arr
#
#     # for reverse-complement sequences, the 1D output is flipped
#     for i in range(half):
#         output_expression[i] = np.flip(output_expression[i], axis=1)
#         if p.species[head_id] == "hg38":
#             output_epigenome[i] = np.flip(output_epigenome[i], axis=1)
#             output_conservation[i] = np.flip(output_conservation[i], axis=1)
#
#     all_outputs = {"our_expression": output_expression}
#     if p.species[head_id] == "hg38":
#         all_outputs["our_epigenome"] = output_epigenome
#         all_outputs["our_conservation"] = output_conservation
#         if hic_num > 0:
#             all_outputs["our_hic"] = output_hic
#     # Cut off the test TSS layer
#     input_sequences = input_sequences[:, :, :-1]
#     print(input_sequences.shape)
#
#     gc.collect()
#     print_memory()
#     for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
#                              key=lambda x: -x[1])[:10]:
#         print("{:>30}: {:>8}".format(name, cm.get_human_readable(size)))
#
#     print(datetime.now().strftime('[%H:%M:%S] ') + "Training")
#
#     if last_proc is not None:
#         print("Waiting")
#         print(mp_q.get())
#         last_proc.join()
#         print("Finished waiting")
#     # joblib.dump([input_sequences, all_outputs], f"{p.pickle_folder}run.gz", compress=3)
#     # argss = joblib.load(f"{p.pickle_folder}run.gz")
#     # p = mp.Process(target=train_step, args=(argss[0][:400], argss[1][:400], argss[2][:400], fit_epochs, head_id,))
#     proc = mp.Process(target=make_model_and_train, args=(
#         heads[p.species[head_id]], p.species[head_id], input_sequences, all_outputs, fit_epochs, hic_num, mp_q,))
#     proc.start()
#     return proc
#
#
# def safe_save(thing, place):
#     joblib.dump(thing, place + "_temp", compress="lz4")
#     if os.path.exists(place):
#         os.remove(place)
#     os.rename(place + "_temp", place)
#
#
# def make_model_and_train(head, head_name, input_sequences, all_outputs, fit_epochs, hic_num, mp_q):
#     print(f"=== Training with head {head_name} ===")
#     import tensorflow as tf
#     # gpus = tf.config.experimental.list_physical_devices('GPU')
#     # for gpu in gpus:
#     #     tf.config.experimental.set_memory_growth(gpu, True)
#     import tensorflow_addons as tfa
#     import model as mo
#     try:
#         if head_name == "hg38":
#             train_data = mo.wrap_for_human_training(input_sequences, all_outputs, p.GLOBAL_BATCH_SIZE)
#         else:
#             train_data = mo.wrap(input_sequences, all_outputs["our_expression"], p.GLOBAL_BATCH_SIZE)
#         del input_sequences
#         del all_outputs
#         gc.collect()
#         strategy = tf.distribute.MultiWorkerMirroredStrategy()
#         with strategy.scope():
#             our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, hic_num if head_name == "hg38" else 0, p.hic_size, head)
#             loss_weights = {}
#             learning_rates = {}
#             weight_decays = {}
#             with open(str(p.script_folder) + "/loss_weights") as f:
#                 for line in f:
#                     (key, weight, lr, wd) = line.split()
#                     if hic_num == 0 and key == "our_hic":
#                         continue
#                     if key != "our_resnet":
#                         loss_weights[key] = float(weight)
#                     learning_rates[key] = float(lr)
#                     weight_decays[key] = float(wd)
#
#             # preparing the main optimizer
#             # tf.keras.optimizers.Adam(learning_rate=learning_rates["our_expression"], clipnorm=0.001)
#             # "our_resnet": tfa.optimizers.AdamW(learning_rate=learning_rates["our_resnet"], weight_decay=0.00001, clipnorm=0.001)
#             optimizers = {
#                 "our_resnet": tfa.optimizers.AdamW(learning_rate=learning_rates["our_resnet"], weight_decay=weight_decays["our_resnet"], clipnorm=0.001),
#                 "our_expression": tfa.optimizers.AdamW(learning_rate=learning_rates["our_expression"], weight_decay=weight_decays["our_expression"], clipnorm=0.001),
#                 "our_epigenome": tfa.optimizers.AdamW(learning_rate=learning_rates["our_epigenome"], weight_decay=weight_decays["our_epigenome"], clipnorm=0.001),
#                 "our_conservation": tfa.optimizers.AdamW(learning_rate=learning_rates["our_conservation"], weight_decay=weight_decays["our_conservation"], clipnorm=0.001)
#                 }
#
#             optimizers_and_layers = [(optimizers["our_resnet"], our_model.get_layer("our_resnet")),
#                                      (optimizers["our_expression"], our_model.get_layer("our_expression"))]
#             if head_name == "hg38":
#                 if hic_num > 0:
#                     optimizers["our_hic"] = tfa.optimizers.AdamW(learning_rate=learning_rates["our_hic"], weight_decay=weight_decays["our_hic"], clipnorm=0.001)
#                     optimizers_and_layers.append((optimizers["our_hic"], our_model.get_layer("our_hic")))
#                 optimizers_and_layers.append((optimizers["our_epigenome"], our_model.get_layer("our_epigenome")))
#                 optimizers_and_layers.append((optimizers["our_conservation"], our_model.get_layer("our_conservation")))
#             optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
#             # loading the loss weights and compiling the model
#             if head_name == "hg38":
#                 losses = {
#                     "our_expression": mo.fast_mse,
#                     "our_epigenome": "mse",
#                     "our_conservation": "mse",
#                 }
#                 if hic_num > 0:
#                     losses["our_hic"] = "mse"
#                 our_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)
#             else:
#                 our_model.compile(optimizer=optimizer, loss="mse")
#             # need to init the optimizers
#             our_model.fit(train_data, steps_per_epoch=1, epochs=1)
#             # loading the previous optimizer weights
#             if os.path.exists(p.model_path + "_opt_expression_" + head_name):
#                 print("loading expression optimizer")
#                 optimizers["our_expression"].set_weights(joblib.load(p.model_path + "_opt_expression_" + head_name))
#                 optimizers["our_expression"].learning_rate.assign(learning_rates["our_expression"])
#             if os.path.exists(p.model_path + "_opt_resnet"):
#                 print("loading resnet optimizer")
#                 optimizers["our_resnet"].set_weights(joblib.load(p.model_path + "_opt_resnet"))
#                 optimizers["our_resnet"].learning_rate.assign(learning_rates["our_resnet"])
#             if head_name == "hg38":
#                 if hic_num > 0 and os.path.exists(p.model_path + "_opt_hic"):
#                     print("loading hic optimizer")
#                     optimizers["our_hic"].set_weights(joblib.load(p.model_path + "_opt_hic"))
#                     optimizers["our_hic"].learning_rate.assign(learning_rates["our_hic"])
#
#                 if os.path.exists(p.model_path + "_opt_epigenome"):
#                     print("loading epigenome optimizer")
#                     optimizers["our_epigenome"].set_weights(joblib.load(p.model_path + "_opt_epigenome"))
#                     optimizers["our_epigenome"].learning_rate.assign(learning_rates["our_epigenome"])
#
#                 if os.path.exists(p.model_path + "_opt_conservation"):
#                     print("loading conservation optimizer")
#                     optimizers["our_conservation"].set_weights(joblib.load(p.model_path + "_opt_conservation"))
#                     optimizers["our_conservation"].learning_rate.assign(learning_rates["our_conservation"])
#             # loading model weights
#             if os.path.exists(p.model_path + "_res"):
#                 our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
#             if os.path.exists(p.model_path + "_expression_" + head_name):
#                 our_model.get_layer("our_expression").set_weights(
#                     joblib.load(p.model_path + "_expression_" + head_name))
#             if head_name == "hg38" and os.path.exists(p.model_path + "_epigenome"):
#                 our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_path + "_epigenome"))
#             if head_name == "hg38" and hic_num > 0 and os.path.exists(p.model_path + "_hic"):
#                 our_model.get_layer("our_hic").set_weights(joblib.load(p.model_path + "_hic"))
#             if head_name == "hg38" and os.path.exists(p.model_path + "_conservation"):
#                 our_model.get_layer("our_conservation").set_weights(
#                     joblib.load(p.model_path + "_conservation"))
#     except:
#         traceback.print_exc()
#         print(datetime.now().strftime('[%H:%M:%S] ') + "Error while compiling.")
#         mp_q.put(None)
#         return None
#
#     try:
#         our_model.fit(train_data, epochs=fit_epochs, batch_size=p.GLOBAL_BATCH_SIZE)
#         del train_data
#         gc.collect()
#         safe_save(our_model.get_layer("our_resnet").get_weights(), p.model_path + "_res")
#         safe_save(optimizers["our_resnet"].get_weights(), p.model_path + "_opt_resnet")
#         safe_save(our_model.get_layer("our_expression").get_weights(), p.model_path + "_expression_" + head_name)
#         safe_save(optimizers["our_expression"].get_weights(), p.model_path + "_opt_expression_" + head_name)
#         if head_name == "hg38":
#             if hic_num > 0:
#                 safe_save(optimizers["our_hic"].get_weights(), p.model_path + "_opt_hic")
#                 safe_save(our_model.get_layer("our_hic").get_weights(), p.model_path + "_hic")
#             safe_save(optimizers["our_epigenome"].get_weights(), p.model_path + "_opt_epigenome")
#             safe_save(our_model.get_layer("our_epigenome").get_weights(), p.model_path + "_epigenome")
#             safe_save(optimizers["our_conservation"].get_weights(), p.model_path + "_opt_conservation")
#             safe_save(our_model.get_layer("our_conservation").get_weights(), p.model_path + "_conservation")
#     except:
#         traceback.print_exc()
#         print(datetime.now().strftime('[%H:%M:%S] ') + "Error while training.")
#         mp_q.put(None)
#         return None
#
#     print(datetime.now().strftime('[%H:%M:%S] ') + "Epoch finished. ")
#     mp_q.put(None)
#     return None
#
#
# def check_perf(mp_q):
#     print(datetime.now().strftime('[%H:%M:%S] ') + "Evaluating")
#     import tensorflow as tf
#     import model as mo
#     one_hot = joblib.load(f"{p.pickle_folder}hg38_one_hot.gz")
#     try:
#         strategy = tf.distribute.MultiWorkerMirroredStrategy()
#         with strategy.scope():
#             our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, 0, p.hic_size, heads["hg38"])
#             our_model.get_layer("our_resnet").set_weights(joblib.load(p.model_path + "_res"))
#             our_model.get_layer("our_expression").set_weights(joblib.load(p.model_path + "_expression_hg38"))
#             our_model.get_layer("our_epigenome").set_weights(joblib.load(p.model_path + "_epigenome"))
#             our_model.get_layer("our_conservation").set_weights(joblib.load(p.model_path + "_conservation"))
#         train_eval_chr = "chr1"
#         train_eval_chr_info = []
#         for info in train_info:
#             if info[0] == train_eval_chr:  # and 139615843 < info[1] < 141668489
#                 train_eval_chr_info.append(info)
#         print(f"Training set {len(train_eval_chr_info)}")
#         training_spearman = evaluation.eval_perf(p, our_model, heads["hg38"], train_eval_chr_info,
#                                                  False, current_epoch, "train", one_hot)
#         # training_spearman = 0
#         test_eval_chr_info = []
#         for info in test_info:
#             # if info[0] == "chr11":
#             test_eval_chr_info.append(info)
#         print(f"Test set {len(test_eval_chr_info)}")
#         test_spearman = evaluation.eval_perf(p, our_model, heads["hg38"], test_eval_chr_info,
#                                              False, current_epoch, "test", one_hot)
#         with open(p.model_name + "_history.csv", "a+") as myfile:
#             myfile.write(f"{training_spearman},{test_spearman}")
#             myfile.write("\n")
#     except Exception as e:
#         print(e)
#         traceback.print_exc()
#         print(datetime.now().strftime('[%H:%M:%S] ') + "Problem during evaluation")
#     mp_q.put(None)
#
#
# def print_memory():
#     mem = psutil.virtual_memory()
#     print(f"used: {cm.get_human_readable(mem.used)} available: {cm.get_human_readable(mem.available)}")
#
#
# def change_seq(x):
#     return cm.rev_comp(x)
#
#
# last_proc = None
# p = MainParams()
# dry_run_regions = {}
# if __name__ == '__main__':
#     # import ray
#     # ray.init()
#     train_info, test_info, protein_coding = parser.parse_sequences(p)
#     if Path(f"{p.pickle_folder}track_names_col.gz").is_file():
#         track_names_col = joblib.load(f"{p.pickle_folder}track_names_col.gz")
#     else:
#         track_names_col = parser.parse_tracks(p)
#
#     if Path(f"{p.pickle_folder}heads.gz").is_file():
#         heads = joblib.load(f"{p.pickle_folder}heads.gz")
#     else:
#         heads = {}
#         for specie in p.species:
#             shuffled_tracks = random.sample(track_names_col[specie], len(track_names_col[specie]))
#             if specie == "hg38":
#                 meta = pd.read_csv("data/ML_all_track.metadata.2022053017.tsv", sep="\t")
#                 new_head = {"expression": [x for x in shuffled_tracks if
#                                            meta.loc[meta['file_name'] == x].iloc[0]["technology"].startswith(
#                                                ("CAGE", "RAMPAGE", "NETCAGE"))],
#                             "epigenome": [x for x in shuffled_tracks if
#                                           meta.loc[meta['file_name'] == x].iloc[0]["technology"].startswith(
#                                               ("DNase", "ATAC", "Histone_ChIP", "TF_ChIP"))],
#                             "conservation": [x for x in shuffled_tracks if
#                                              meta.loc[meta['file_name'] == x].iloc[0]["value"].startswith(
#                                                  "conservation")]}
#             else:
#                 new_head = shuffled_tracks
#             heads[specie] = new_head
#         joblib.dump(heads, f"{p.pickle_folder}heads.gz", compress="lz4")
#
#     for head_key in heads.keys():
#         if head_key == "hg38":
#             for human_key in heads[head_key]:
#                 print(f"Number of tracks in head {head_key} {human_key}: {len(heads[head_key][human_key])}")
#         else:
#             print(f"Number of tracks in head {head_key}: {len(heads[head_key])}")
#
#     # hic_keys = pd.read_csv("data/good_hic.tsv", sep="\t", header=None).iloc[:, 0]
#     hic_keys = []
#     hic_num = len(hic_keys)
#     print(f"hic {hic_num}")
#
#     # import model as mo
#     # our_model = mo.make_model(p.input_size, p.num_features, p.num_bins, hic_num, p.hic_size, heads["hg38"])
#
#     # load_old_weights()
#     # exit()
#     mp_q = mp.Queue()
#     print("Training starting")
#     start_epoch = 0
#     fit_epochs = 4
#     try:
#         for current_epoch in range(start_epoch, p.num_epochs, 1):
#             head_id = 0
#             # with open(str(p.script_folder) + "/../step_num") as f:
#             #     p.STEPS_PER_EPOCH = int(f.read())
#             # if current_epoch % 2 == 0:
#             #     head_id = 0
#             # else:
#             #     head_id = 1 + (current_epoch - math.ceil(current_epoch / 2)) % (len(heads) - 1)
#             # if head_id == 0:
#             #     p.STEPS_PER_EPOCH = 400
#             #     fit_epochs = 4
#             # else:
#             #     p.STEPS_PER_EPOCH = 600
#             #     fit_epochs = 2
#
#             # check_perf(mp_q)
#             # exit()
#             last_proc = get_data_and_train(last_proc, fit_epochs, head_id)
#             if current_epoch % 10 == 0:  # and current_epoch != 0:
#                 print("Eval epoch")
#                 print(mp_q.get())
#                 last_proc.join()
#                 last_proc = None
#                 proc = mp.Process(target=check_perf, args=(mp_q,))
#                 proc.start()
#                 print(mp_q.get())
#                 proc.join()
#     except Exception as e:
#         traceback.print_exc()
#
# # for key in dry_run_regions.keys():
# #     with open(f"dry/{key}_dry_test.bed", "w") as text_file:
# #         text_file.write("\n".join(dry_run_regions[key]))
