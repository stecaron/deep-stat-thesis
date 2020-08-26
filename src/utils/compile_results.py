import glob
import pandas
import numpy
import os
import math


RESULTS_FOLDER = "results/2020-08-05/"

if os.path.exists(RESULTS_FOLDER+"compiled_results.csv"):
    os.remove(RESULTS_FOLDER+"compiled_results.csv")

results_files = glob.glob(RESULTS_FOLDER + "*.csv")

cols = ["experiment", "mean_f1_score", "sd_f1_score", "mean_auc", "sd_auc"]
df_compiled = pandas.DataFrame(columns=cols)

for file in results_files:
    temp_results = pandas.read_csv(file)

    experiment = file.split("/")[2].split(".")[0]
    mean_f1_score = numpy.mean(temp_results["f1_score"])
    sd_f1_score = numpy.std(temp_results["f1_score"])
    mean_auc = numpy.mean(temp_results["auc"])
    sd_auc = numpy.std(temp_results["auc"])

    if math.isnan(mean_auc):
        mean_auc = numpy.array(0)
        sd_auc = numpy.array(0)

    df_temp = pandas.DataFrame(
        numpy.concatenate((
            numpy.array(experiment).reshape(1),
            mean_f1_score.reshape(1),
            sd_f1_score.reshape(1),
            mean_auc.reshape(1),
            sd_auc.reshape(1)
        )).reshape(1, -1), columns=cols
    )

    df_compiled = df_compiled.append(df_temp, ignore_index=True)

df_compiled.to_csv(RESULTS_FOLDER+"compiled_results.csv")