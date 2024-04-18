<div align="center">
   <a href="https://github.com/liyinred/Cluster_Wenhao/blob/main/Cluster_dif.md" target="_blank">knn and kmeans difference</a>
</div>

## FAISS的IVF-PQ算法基于这个做谱图的聚类识别的具体流程
1. 生成谱图数据：利用 NumPy 库生成了包含 2000 个谱图（每个谱图有 200 个特征）的示例数据集。这个步骤是为了模拟真实的谱图数据，用于后续的聚类和相似性搜索。
2. K-means 聚类：利用 Faiss 库中的 K-means 算法将谱图数据进行聚类。这一步骤的目的是将数据集分成 100 个聚类中心，从而减少数据维度并提高搜索效率。
3. Product Quantization (PQ) 训练：定义了 PQ 的参数，包括子量化器数量和每个子量化器的比特数。PQ 是一种用于向量量化的技术，通过将高维向量映射到低维空间来减少存储和计算开销。在这里，使用之前得到的聚类中心数据对 PQ 进行了训练。
4. 创建 IVF 索引：创建了一个 Inverted File (IVF) 索引，它是一种倒排索引结构，利用了聚类中心和量化技术来实现快速的相似性搜索。在这里，使用 FlatL2 作为基础量化器，并使用之前训练得到的 PQ 进行量化，同时指定了 IVF 中的聚类数量和子量化器参数。
5. 添加数据到 IVF 索引：将之前训练得到的 Product Quantizer 和聚类中心添加到 IVF 索引中，这样 IVF 索引就包含了用于相似性搜索的关键数据结构。
6. 定义查询谱图：生成了 5 个随机的查询谱图，用于后续的相似性搜索操作。
7. 查询和识别：对每个查询谱图进行查询操作，找到最接近的 3 个邻居谱图，并计算它们与查询谱图之间的相似性分数。这个过程是通过 IVF 索引和 PQ 技术实现的，可以高效地找到相似的谱图数据。
```python
import numpy as np
import faiss

# 创建示例谱图数据
np.random.seed(123)
n_spectra = 2000
n_features = 200
spectra_data = np.random.rand(n_spectra, n_features).astype(np.float32)

# 定义聚类参数
n_clusters = 100

# 使用 K-means 聚类将谱图数据分成多个聚类中心
kmeans = faiss.Kmeans(n_features, n_clusters)
kmeans.train(spectra_data)
cluster_centers = kmeans.centroids  # 获取聚类中心向量

# 定义 Product Quantization (PQ) 参数
n_subquantizers = 8
n_bits_per_subquantizer = 4

# 创建并训练 Product Quantizer
pq = faiss.IndexPQ(n_features, n_subquantizers, n_bits_per_subquantizer)
pq.train(cluster_centers)
pq.add(cluster_centers)

# 创建 IVF 索引
quantizer = faiss.IndexFlatL2(n_features)
index = faiss.IndexIVFPQ(quantizer, n_features, n_clusters, n_subquantizers, n_bits_per_subquantizer)

# 将 Product Quantizer 和聚类中心添加到 IVF 索引中
index.train(cluster_centers)
index.add(cluster_centers)

# 定义查询谱图
n_query = 5
query_spectra = np.random.rand(n_query, n_features).astype(np.float32)

# 进行谱图聚类识别并获取 top K 最近邻
k = 3
distances, indices = index.search(query_spectra, k)

# 打印识别结果
for i in range(n_query):
    print(f"Query spectrum {i}:")
    valid_clusters_found = False
    for j in range(k):
        cluster_index = indices[i][j]
        similarity_score = distances[i][j]
        if cluster_index != -1:
            valid_clusters_found = True
            print(f"  Neighbor {j + 1}: Cluster {cluster_index} (Similarity Score: {similarity_score:.4f})")
    if not valid_clusters_found:
        print(f"  No valid clusters found for Query spectrum {i}")
```




