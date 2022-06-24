# get the data set

path = "/content/drive/My Drive/pyspatktp/my_file.csv"

df = spark.read.csv(path, header = True, sep = ';', inferSchema = True)

#df2 = spark.read.option(delimiter=';').csv(path)

df.printSchema()
df.show(4,False)
#df.count()

# Remove NA values

df = df.na.drop()

# get columns names

df.columns

# take the columns you want to featur

feat_cols = ['C1',
 'C2']

# assemble the columns

assembler = VectorAssembler(inputCols = feat_cols, outputCol = 'features')
final_df = assembler.transform(df)

# norm the data between 0 and 1

from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol='features', 
                        outputCol='scaled_feat',
                        withStd = True,
                        withMean = False)

scaled_model = scaler.fit(final_df)

cluster_df = scaled_model.transform(final_df)

# k-means classification ( here i do it between 2 and 11 because i worked with a data set of 10 columns)

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

eval = ClusteringEvaluator(predictionCol="prediction",
                           featuresCol="scaled_feat",
                           metricName="silhouette",
                           distanceMeasure="squaredEuclidean")

silhouette_score = []
print("""
Silhoutte Scores for K Mean Clustering
======================================
Model\tScore\t
=====\t=====\t
""")
for k in range(2,11):
  kmeans_algo = KMeans(featuresCol='scaled_feat',k=k)
  kmeans_fit = kmeans_algo.fit(cluster_df)
  output = kmeans_fit.transform(cluster_df)
  score = eval.evaluate(output)
  silhouette_score.append(score)
  print(f"K{k}\t{round(score,2)}\t")
  
# We plot the results for the diffenret number of k-means

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(10,10))
ax.plot(range(2,11), silhouette_score)
ax.set_xlabel("K")
ax.set_ylabel("Score");
