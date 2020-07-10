###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
import random
import prince
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def pca_results(score,coeff,labels=None):
    plt.figure(figsize=(20,10))
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley)

    for i in range(n):
        plt.arrow(1.3, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(1.3 + coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(0.5,2)
    plt.ylim(-1,1)
    plt.xlabel("COMPONENTE{}".format(1))
    plt.ylabel("COMPONENTE{}".format(2))
    plt.grid()

def pca_results2(df, famd):
	'''
	Create a DataFrame of the PCA results
	Includes dimension feature weights and explained variance
	Visualizes the PCA results
	'''

	# Dimension indexing
	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,famd.n_components+1)]

	# PCA components
	components = pd.DataFrame(np.round(np.array(famd.column_correlations(df)), 4), columns = list(df.keys()))
	components.index = dimensions

	# PCA explained variance
	ratios = famd.explained_variance_ratio_.reshape(famd.n_components, 1)
	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
	variance_ratios.index = dimensions

	# Create a bar plot visualization
	fig, ax = plt.subplots(figsize = (14,8))

	# Plot the feature weights as a function of the components
	components.plot(ax = ax, kind = 'bar');
	ax.set_ylabel("Feature Weights")
	ax.set_xticklabels(dimensions, rotation=0)


	# Display the explained variance ratios
	# for i, ev in enumerate(fdma.explained_variance_ratio_):
	# 	ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

	# Return a concatenated DataFrame
	return pd.concat([variance_ratios, components], axis = 1)
def cluster_results(reduced_data, preds, centers, pca_samples):
	'''
	Visualizes the PCA-reduced cluster data in two dimensions
	Adds cues for cluster centers and student-selected sample data
	'''

	predictions = pd.DataFrame(preds, columns = ['Cluster'])
	plot_data = pd.concat([predictions, reduced_data], axis = 1)

	# Generate the cluster plot
	fig, ax = plt.subplots(figsize = (14,8))

	# Color map
	cmap = cm.get_cmap('gist_rainbow')

	# Color the points based on assigned cluster
	for i, cluster in plot_data.groupby('Cluster'):   
	    cluster.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \
	                 color = cmap((i)*1.0/(len(centers)-1)), label = 'Cluster %i'%(i+1), s=30);

	# Plot centers with indicators
	for i, c in enumerate(centers):
	    ax.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black', \
	               alpha = 1, linewidth = 2, marker = 'o', s=200);
	    ax.scatter(x = c[0], y = c[1], marker='$%d$'%(i+1), alpha = 1, s=100);

	# Plot transformed sample points 
	# ax.scatter(x = pca_samples[:,0], y = pca_samples[:,1], \
	#            s = 150, linewidth = 4, color = 'black', marker = 'x');

	# Set plot title
	ax.set_title("Cluster Learning on PCA-Reduced Data - Centroids Marked by Number\n");


def biplot(good_data, reduced_data, famd):
    '''
    Produce a biplot that shows a scatterplot of the reduced
    data and the projections of the original features.
    
    good_data: original data, before transformation.
               Needs to be a pandas dataframe with valid column names
    reduced_data: the reduced data (the first two dimensions are plotted)
    pca: pca object that contains the components_ attribute

    return: a matplotlib AxesSubplot object (for any additional customization)
    
    This procedure is inspired by the script:
    https://github.com/teddyroland/python-biplot
    '''

    fig, ax = plt.subplots(figsize = (14,8))
    # scatterplot of the reduced data    
    ax.scatter(x=reduced_data.loc[:, 'Dimension 1'], y=reduced_data.loc[:, 'Dimension 2'], 
        facecolors='b', edgecolors='b', s=70, alpha=0.5)
    
    feature_vectors = famd.explained_inertia_
    #feature_vectors = pca.components_.T
    #famd.explained_inertia_

    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos = 7.0, 8.0,

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], 
                  head_width=0.2, head_length=0.2, linewidth=2, color='red')
        ax.text(v[0]*text_pos, v[1]*text_pos, good_data.columns[i], color='black', 
                 ha='center', va='center', fontsize=18)

    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("PC plane with original feature projections.", fontsize=16);
    return ax
    

def channel_results(reduced_data, outliers, pca_samples):
	'''
	Visualizes the PCA-reduced cluster data in two dimensions using the full dataset
	Data is labeled by "Channel" and cues added for student-selected sample data
	'''

	# Check that the dataset is loadable
	try:
	    full_data = pd.read_csv("customers.csv")
	except:
	    print("Dataset could not be loaded. Is the file missing?")       
	    return False

	# Create the Channel DataFrame
	channel = pd.DataFrame(full_data['Channel'], columns = ['Channel'])
	channel = channel.drop(channel.index[outliers]).reset_index(drop = True)
	labeled = pd.concat([reduced_data, channel], axis = 1)
	
	# Generate the cluster plot
	fig, ax = plt.subplots(figsize = (14,8))

	# Color map
	cmap = cm.get_cmap('gist_rainbow')

	# Color the points based on assigned Channel
	labels = ['Hotel/Restaurant/Cafe', 'Retailer']
	grouped = labeled.groupby('Channel')
	for i, channel in grouped:   
	    channel.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \
	                 color = cmap((i-1)*1.0/2), label = labels[i-1], s=30);
	    
	# Plot transformed sample points   
	for i, sample in enumerate(pca_samples):
		ax.scatter(x = sample[0], y = sample[1], \
	           s = 200, linewidth = 3, color = 'black', marker = 'o', facecolors = 'none');
		ax.scatter(x = sample[0]+0.25, y = sample[1]+0.3, marker='$%d$'%(i), alpha = 1, s=125);

	# Set plot title
	ax.set_title("PCA-Reduced Data Labeled by 'Channel'\nTransformed Sample Data Circled");

	###########################################
#This is modified code from original code at: 
#http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py

def silhouette_score_graph(range_n_clusters,algorith,values):
    
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(10, 4)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(values.index) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        if algorith=='KNN':        
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(values)  
                 
        elif  algorith=='GMM':
            clusterer = GaussianMixture(n_components=n_clusters, random_state=10).fit(values)
            cluster_labels = clusterer.predict(values)
        

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(values, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", round(silhouette_avg,4))

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(values, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(values['Dimension 1'], values['Dimension 2'], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
        
        # Labeling the clusters
        if algorith=='KNN':            
            centers = clusterer.cluster_centers_
            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        elif  algorith=='GMM':
            centers = clusterer.means_
            plt.suptitle(("Silhouette analysis for Gaussian Mixture Model clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
           
       
        
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st component")
        ax2.set_ylabel("Feature space for the 2nd component")

        plt.show()

def Function_Clustering(algorith,values):
    # Loop through clusters
    results = []
    for n_clusters in list_n_clusters:
        
        if algorith=='KNN':        

            clusterer = KMeans(n_clusters=n_clusters,random_state=10).fit(values)     
            centers = clusterer.cluster_centers_
        
        elif  algorith=='GMM':
        
             
            clusterer = GaussianMixture(n_components=n_clusters, random_state=10).fit(values)                
            centers = clusterer.means_
        

         
        preds = clusterer.predict(values)        
        #sample_preds = clusterer.predict(pca_samples)        
        score = silhouette_score(values, preds, metric='euclidean')
        results.append({'Clusters':n_clusters,'silhouette_score':round(score,4)})
   
    return results

def show_exploration(df):
    clr  =  [ '#12efff' ,  '#abc222' ,  '#00ef00',  '#ffa700' ,  '#d62d20' ,  '#a200ff']
    
    plt.subplots(figsize=(15,10))
    plt.figure(1)


    ax1=plt.subplot(1,3,1)
    plt.xticks(rotation=45)
    ax1.set_title('ESTUDIOS')
    df['ES'].hist (figsize=(18,5))

    ax1=plt.subplot(1,3,2)
    plt.xticks(rotation=45)
    ax1.set_title('SEXO')
    df['SEXO'].hist (figsize=(18,5))

    ax1=plt.subplot(1,3,3)
    plt.xticks(rotation=45)
    ax1.set_title('TIPOTRABAJO')
    df['TT'].hist(figsize=(18,5))

    
    plt.figure(2)


    ax1=plt.subplot(2,2,1)
    plt.xticks(rotation=45)
    ax1.set_title('FUNCION DESARROLLADA')
    df['FD'].hist(figsize=(18,5))

    ax1=plt.subplot(2,2,2)
    plt.xticks(rotation=45)
    ax1.set_title('Gestiona Equipos')
    df['GE'].hist(figsize=(18,5))



    plt.subplots(figsize=(15,10))
    plt.figure(3)
    plt.suptitle(u'Edad. \n' 'Note: Solid vertical line represents mean, dashed line represents median.')

    ax1=plt.subplot(3,2,1)
    sns.distplot(df['EDAD'], color =clr[5])
    plt.axvline(df['EDAD'].mean(), color='#000000', linestyle='solid', linewidth=1)
    plt.axvline(df['EDAD'].median(), color='#000000', linestyle='dashed', linewidth=1)
    ax1.set_title('Distribución de la edad')

    ax1=plt.subplot(3,2,2)
    _= sns.boxplot(data=df['EDAD'], orient='h', palette=clr,ax=ax1)

    #Values range
    ax2=plt.subplot(3,2,3)
    _ = sns.barplot(data=df['EDAD'], palette=clr,ax=ax2)

def show_exploration_segment(df):
    clr  =  [ '#12efff' ,  '#abc222' ,  '#00ef00',  '#ffa700' ,  '#d62d20' ,  '#a200ff']
    
    plt.subplots(figsize=(15,10))
    plt.figure(1)


    ax1=plt.subplot(1,3,1)
    plt.xticks(rotation=45)
    ax1.set_title('SEXO')
    df['SEXO'].hist (figsize=(18,5))

    ax1=plt.subplot(1,3,2)
    plt.xticks(rotation=45)
    ax1.set_title('ESTUDIOS')
    df['estudios'].hist (figsize=(18,5))

    ax1=plt.subplot(1,3,3)
    plt.xticks(rotation=45)
    ax1.set_title('TIPOTRABAJO')
    df['TIPOTRABAJO'].hist(figsize=(18,5))

    
    plt.figure(2)


    ax1=plt.subplot(2,3,1)
    plt.xticks(rotation=45)
    ax1.set_title('FUNCION DESARROLLADA')
    df['FUNCIONDESARROLLADA'].hist(figsize=(18,5))

    ax1=plt.subplot(2,3,2)
    plt.xticks(rotation=45)
    ax1.set_title('Gestiona Equipos')
    df['GestionaEquipos'].hist(figsize=(18,5))

    ax1=plt.subplot(2,3,3)
    plt.xticks(rotation=45)
    ax1.set_title('Numero Personas')
    df['NumeroPersonas'].hist(figsize=(18,5))

    # plt.subplots(figsize=(15,10))    
    # plt.figure(3)
    # plt.suptitle(u'Edad. \n' 'Note: Solid vertical line represents mean, dashed line represents median.')

    # ax1=plt.subplot(3,2,1)
    # sns.distplot(df['EDAD'], color =clr[5])
    # plt.axvline(df['EDAD'].mean(), color='#000000', linestyle='solid', linewidth=1)
    # plt.axvline(df['EDAD'].median(), color='#000000', linestyle='dashed', linewidth=1)
    # ax1.set_title('Distribución de la edad')

    # ax1=plt.subplot(3,2,2)
    # _= sns.boxplot(data=df['EDAD'], orient='h', palette=clr,ax=ax1)

    # #Values range
    # ax2=plt.subplot(3,2,3)
    # _ = sns.barplot(data=df['EDAD'], palette=clr,ax=ax2)


def show_exploration_segment_comportamiento(df,dfti=None):
    clr  =  [ '#12efff' ,  '#abc222' ,  '#00ef00',  '#ffa700' ,  '#d62d20' ,  '#a200ff']
    #colunmas= ['Segmento','DispuestoPagar','coaching','Pagado','TemaInteres','sessiones']
    plt.subplots(figsize=(15,10))
    plt.figure(1)


    ax1=plt.subplot(1,3,1)
    plt.xticks(rotation=45)
    ax1.set_title('Dispuesto Pagar')
    df['DispuestoPagar'].hist (figsize=(18,5))

    ax1=plt.subplot(1,3,2)
    plt.xticks(rotation=45)
    ax1.set_title('Coaching')
    df['coaching'].hist (figsize=(18,5))

    ax1=plt.subplot(1,3,3)
    plt.xticks(rotation=45)
    ax1.set_title('Pagado')
    df['Pagado'].hist(figsize=(18,5))

    
    plt.figure(2)

        
    ax1=plt.subplot(2,2,1)
    plt.xticks(rotation=90)
    ax1.set_title('Tema Interes')
    dfti['TemaInteres'].hist(figsize=(18,5))

    ax1=plt.subplot(2,2,2)
    plt.xticks(rotation=45)
    ax1.set_title('sessiones')
    df['sessiones'].hist(figsize=(18,5))



def show_sample_exploration(df,samples,indices):

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

def show_components(famd):
    dset = pd.DataFrame()
    dset['famd'] = range(1,7)
    dset['eigenvalue'] = pd.DataFrame(famd.eigenvalues_)
    plt.figure(figsize=(18,6))
    ax1 = plt.subplot(1, 2, 1)
    sns.lineplot(x='famd', y='eigenvalue', marker="o", data=dset)
    plt.ylabel('Eigenvalue', fontsize=16)
    plt.xlabel('Principal Component', fontsize=16)

    ax1 = plt.subplot(1, 2, 2)
    dset['vari'] = pd.DataFrame(famd.explained_inertia_)

    graph = sns.barplot(x='famd', y='vari', data=dset)
    for p in graph.patches:
        graph.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')
    plt.ylabel('Proportion', fontsize=18)
    plt.xlabel('Principal Component', fontsize=18)

def show_pather(df,famd, slabel=False):

    scatter = pd.DataFrame(famd.column_correlations(df)).reset_index()
    plt.figure(figsize=(25,10))
    ax = sns.scatterplot(x=0, y=1, data=scatter)
    ax.set(ylim=(-1, 1), xlim=(-1,1.5))
    
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            texto= str(point['val']).split("_")
            valor = texto[-1]        
            #ax.text(point['x']+.02, point['y'],point['val'] )
            ax.text(point['x']+.02, point['y'],valor )
    if slabel:
        label_point(scatter[0], scatter[1], scatter['index'], plt.gca())  

    plt.axvline(-0.5, ls='--')
    plt.axhline(0, ls='--')

    plt.title('Pattern Plot of Component 2 by Component 1', fontsize=18)
    plt.xlabel('Component 1 (41%)', fontsize=10)
    plt.ylabel('Component 2 (13%)', fontsize=10)
    plt.show()

valueCounts = {}
def CountAll(df):
    global all_columns, nanCounts, valueCounts
    all_columns = list(df)
    nanCounts = df.isnull().sum()
    for x in all_columns:
        valueCounts[x] = df[x].value_counts()        


def Fill_NaNs_Categorical(df,col):     
    """Calculating probability and expected value."""
    proportion = np.array(valueCounts[col].values) / valueCounts[col].sum() * nanCounts[col]
    proportion = np.around(proportion).astype('int')
    
    """Adjusting proportion."""
    diff = int(nanCounts[col] - np.sum(proportion))
    if diff > 0:
        for x in range(diff):
            idx = random.randint(0, len(proportion) - 1)
            proportion[idx] =  proportion[idx] + 1
    else:
        diff = -diff
        while(diff != 0):
            idx = random.randint(0, len(proportion) - 1)
            if proportion[idx] > 0:
                proportion[idx] =  proportion[idx] - 1
                diff = diff - 1
        
    """Filling NaNs."""
    nan_indexes = df[df[col].isnull()].index.tolist() 
    for x in range(len(proportion)):
        if proportion[x] > 0:
            random_subset = random.sample(population = nan_indexes, k = proportion[x])
            df.loc[random_subset, col] = valueCounts[col].keys()[x]
            nan_indexes = list(set(nan_indexes) - set(random_subset))

def agregarNuevasVariables(df_aux):
    dffp= pd.read_csv('DispuestoPagar.csv')
    df_aux['DispuestoPagar']= dffp['DispuestoPagar']


    dfc= pd.read_csv('coaching.csv',sep=';')
    df_aux['coaching'] = dfc['coaching']

    dfpp= pd.read_csv('PagadoSession2.csv')
    df_aux['Pagado']= dfpp['PagadoSesion']
    df_aux['Pagado'][0]=df_aux['Pagado'][64]

    dfti= pd.read_csv('TemaInteres2.csv', encoding='utf-8')
    df_aux['TemaInteres']= dfti['TemaInteres']

    dfss= pd.read_csv('session.csv')
    df_aux['sessiones']= dfss['sessiones']
    df_aux['sessiones'][0]=df_aux['sessiones'][65]


    #dfex= pd.read_csv('ExpectativasHaciaCoach.csv',sep=';')
    #dfs['ExpectativasHaciaCoach']= dfex['ExpectativasHaciaCoach']
    #dfs['sessiones'][0]=dfs['sessiones'][65]

    return df_aux.copy()


def  plot_comportamiento_segmento(df,key):
    int_col = ['Segmento','DispuestoPagar','coaching','Pagado','TemaInteres','sessiones']
    plt.figure(figsize=(8,6))
    if key=='DispuestoPagar':
        values = ['Entre 10-40', 'Entre 45-70', 'Entre 75-100']

    if key=='coaching':
        values = ['Si de forma puntual', 'no me lo he planteado', 'pensado','los uso con frecuencia']

    if key=='Pagado':
        values = ['Entre 10-40', 'Entre 45-70', 'Entre 75-100','Mayor de 100']
    
    if key=='sessiones':
        values = ['Mayor de 10', 'Entre 4-6.', 'Entre 1-3.','Entre 7-10.']

    if key=='TemaInteres':
        int_col = ['Segment']
        values = ['Gestión de equipos', 'Habilidades de comunicación', \
        'Gestión del tiempo','Motivación e inspiración',\
            'Relaciones de pareja y familia','Pensamiento/Planificación estratégica', \
            'Autoconfianza','Desarrollo de cualidades personales','Fe y espiritualidad',\
            'Desarrollo personal','Conflicto laboral','Liderazgo y confianza']

    frame = pd.DataFrame(index = np.arange(len(values)), columns=(key,'Seg0','Seg1'))
    for i, value in enumerate(values):
        if key=='TemaInteres':
            frame.loc[i] = [value, \
                len(df['Segmento'][(df['Segmento'] == 0) & (df[key] == value)]), \
                len(df['Segmento'][(df['Segmento'] == 1) & (df[key] == value)])]
        else:
            frame.loc[i] = [value, \
                len(df[int_col][(df[int_col]['Segmento'] == 0) & (df[int_col][key] == value)]), \
                len(df[int_col][(df[int_col]['Segmento'] == 1) & (df[int_col][key] == value)])]

    bar_width = 0.4
    for i in np.arange(len(frame)):
                nonsurv_bar = plt.bar(i-bar_width, frame.loc[i]['Seg0'], width = bar_width, color = 'r')
                surv_bar = plt.bar(i, frame.loc[i]['Seg1'], width = bar_width, color = 'g')

                plt.xticks(np.arange(len(frame)), values)
                plt.xticks(rotation=90)
                plt.legend((nonsurv_bar[0], surv_bar[0]),('Segemto 0', 'Segmento 1'), framealpha = 0.8)
    
    
    

def fix_temaInteres(df):
    dfTemaInteres= pd.DataFrame(columns=['Segmento', 'TemaInteres','IndexFrom'])
    for index, row in df.iterrows():    
        arraux= row['TemaInteres'].split('|')
        for i,val in enumerate(arraux):   
            
            #print(val)
            #print(row['Segment'])     
            #print(val)
            #print(index)
            #new_row = pd.Series({'Segment':row['Segment'], 'TemaInteres':val, 'IndexFrom':index})
            #dfaux.append(new_row,ignore_index=True )    

            dfTemaInteres.loc[index] = [row['Segmento'], [val.split('.')][0][0].strip(),index]
        
        
        
    #dfTemaInteres.head()
    return dfTemaInteres

def  plot_comportamientos_segmento(df):
    int_col = ['Segmento','DispuestoPagar','coaching','Pagado','TemaInteres','sessiones']
    plt.figure(figsize=(8,6))
    tframe = pd.DataFrame()
    values = None
    paso=False

    for key in int_col:
        print(key)
        if key=='Segmento':
            paso=False
            print(paso)
            continue
        paso=True
        if key=='DispuestoPagar':
           values = ['Entre 10-40', 'Entre 45-70', 'Entre 75-100']

        if key=='coaching':
            values = ['Si de forma puntual', 'no me lo he planteado', 'pensado','los uso con frecuencia']

        if key=='Pagado':
            values = ['Entre 10-40', 'Entre 45-70', 'Entre 75-100','Mayor de 100']
        
        if key=='sessiones':
            values = ['Mayor de 10', 'Entre 4-6.', 'Entre 1-3.','Entre 7-10.']

        if key=='TemaInteres':
            int_col = ['Segment']
            values = ['Gestión de equipos', 'Habilidades de comunicación', \
            'Gestión del tiempo','Motivación e inspiración',\
                'Relaciones de pareja y familia','Pensamiento/Planificación estratégica', \
                'Autoconfianza','Desarrollo de cualidades personales','Fe y espiritualidad',\
                'Desarrollo personal','Conflicto laboral','Liderazgo y confianza']

        if paso:
            print(paso)
            print(key)
            tframe = pd.DataFrame(index = np.arange(len(values)), columns=(key,'Seg0','Seg1'))
            
            for i, value in enumerate(values):
                if key=='TemaInteres':
                    tframe.loc[i] = [value, \
                        len(df['Segmento'][(df['Segmento'] == 0) & (df[key] == value)]), \
                        len(df['Segmento'][(df['Segmento'] == 1) & (df[key] == value)])]
                else:
                    print(value)
                    tframe.loc[i] = [value, \
                len(df[int_col][(df[int_col]['Segmento'] == 0) & (df[int_col][key] == value)]), \
                len(df[int_col][(df[int_col]['Segmento'] == 1) & (df[int_col][key] == value)])]
        paso=False

        bar_width = 0.4
        plt.subplots(figsize=(15,10))
        plt.figure(1)
        k=1   


        for i in np.arange(len(tframe)):
                    nonsurv_bar = plt.bar(i-bar_width, tframe.loc[i]['Seg0'], width = bar_width, color = 'r')
                    surv_bar = plt.bar(i, tframe.loc[i]['Seg1'], width = bar_width, color = 'g')

                    ax1=plt.subplot(1,3,k)
                    plt.xticks(rotation=45)
                    ax1.set_title(key)                  

                    plt.xticks(np.arange(len(tframe)), values)
                    plt.xticks(rotation=90)
                    plt.legend((nonsurv_bar[0], surv_bar[0]),('Segemto 0', 'Segmento 1'), framealpha = 0.8)

def show_comportamiento(df,segmento):
    colunmas= ['Segmento','DispuestoPagar','coaching','Pagado','TemaInteres','sessiones']
    en_filtro=df['Segmento']==segmento
    df_segment = df[en_filtro][colunmas]
    dfTemaInteres= fix_temaInteres(df_segment)
    show_exploration_segment_comportamiento(df_segment,dfTemaInteres)

