import glob
import pandas
import numpy
import os
import math


RESULTS_FOLDER = "results/2020-10-24/"

if os.path.exists(RESULTS_FOLDER+"compiled_results.csv"):
    os.remove(RESULTS_FOLDER+"compiled_results.csv")

folders = glob.glob(RESULTS_FOLDER+"/*/")

cols = ["experiment", "scenario", "mean_f1_score", "sd_f1_score", "mean_auc", "sd_auc", "mean_precision", "sd_precision", "mean_recall", "sd_recall"]
df_compiled = pandas.DataFrame(columns=cols)
df_temp = pandas.DataFrame(columns=cols)


for folder in folders:

    results_files = glob.glob(folder + "*.csv")
    results_files = [i for i in results_files if i.split("/")[3].split(".")[0].startswith('results_')]

    for file in results_files:
        temp_results = pandas.read_csv(file)

        experiment = file.split("/")[2]
        scenario = file.split("/")[3].split(".")[0].replace("results_", "")
        mean_f1_score = numpy.mean(temp_results["f1_score"])
        sd_f1_score = numpy.std(temp_results["f1_score"])
        mean_auc = numpy.mean(temp_results["auc"])
        sd_auc = numpy.std(temp_results["auc"])
        mean_precision = numpy.mean(temp_results["precision"])
        sd_precision = numpy.std(temp_results["precision"])
        mean_recall = numpy.mean(temp_results["recall"])
        sd_recall = numpy.std(temp_results["recall"])

        if math.isnan(mean_auc):
            mean_auc = numpy.array(0)
            sd_auc = numpy.array(0)

        df_temp = pandas.DataFrame(
            numpy.concatenate((
                numpy.array(experiment).reshape(1),
                numpy.array(scenario).reshape(1),
                mean_f1_score.reshape(1),
                sd_f1_score.reshape(1),
                mean_auc.reshape(1),
                sd_auc.reshape(1),
                mean_precision.reshape(1),
                sd_precision.reshape(1),
                mean_recall.reshape(1),
                sd_recall.reshape(1),
            )).reshape(1, -1), columns=cols
        )
        df_compiled = df_compiled.append(df_temp, ignore_index=True)

df_compiled.to_csv(RESULTS_FOLDER+"compiled_results.csv")