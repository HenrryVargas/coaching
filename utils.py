import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt

def plot_muestra(df,indices):
    ax1 = plt.subplot(1, 2, 1)

    #The means 
    mean_data = df.describe().loc['mean', :]

    #Append means to the samples' data
    samples_bar = samples.append(mean_data)

    #Construct indices
    samples_bar.index = indices + ['mean']

    #Plot bar plot
    samples_bar.plot(kind='bar', figsize=(15,5), ax=ax1)
    ax1.set_title("Samples vs Mean")

    ax2 = plt.subplot(1, 2, 2)

    # percentile ranks of the whole dataset.
    percentiles = df.rank(pct=True)

    # Round it up, and multiply by 100
    percentiles = 100*percentiles.round(decimals=3)

    # Select the indices from the percentiles dataframe
    percentiles = percentiles.iloc[indices]

    # Now, create the heat map
    sns.heatmap(percentiles, vmin=1, vmax=99, ax=ax2, annot=True)
    ax2.set_title("Comparación de los percentiles de la muestra.")