# Cluster_Wenhao
KNN（K-Nearest Neighbors）和K-Means是两种常用的机器学习算法，它们在原理和应用上有着本质的不同。

### KNN（K-最近邻）算法

**原理**：
- KNN是一种监督学习算法，用于分类和回归。
- 对于一个未知类别的样本，KNN算法会在训练集中找到与之最近的K个邻居，然后通过多数投票的方式来预测未知样本的类别（分类问题）或计算邻居的平均值（回归问题）。

**步骤**：
1. 确定邻居的数量K。
2. 计算未知样本与所有训练样本之间的距离。
3. 选择距离最近的K个样本。
4. 对分类问题，选择这K个样本中出现次数最多的类别作为预测结果；对回归问题，计算这K个样本的均值作为预测结果。

**特点**：
- KNN算法简单，不需要训练模型，计算量主要集中在预测阶段。
- K值的选择对结果影响较大。
- 对噪声敏感，需要预处理数据。

### K-Means（K-均值）算法

**原理**：
- K-Means是一种非监督学习算法，主要用于聚类。
- 算法接受一个参数K，然后将数据集中的样本划分为K个聚类，使得每个聚类内部样本之间的相似度尽可能高，而不同聚类之间的相似度尽可能低。

**步骤**：
1. 随机选择K个样本作为初始聚类中心。
2. 对数据集中的每一个样本，计算它与各个聚类中心的距离，并将其归到距离最近的聚类中心所在的类。
3. 根据聚类结果，重新计算每个聚类的中心点。
4. 重复步骤2和3，直到聚类中心不再发生变化或达到迭代次数上限。

**特点**：
- K值的选择对聚类结果影响很大，通常需要通过其他方法来确定K值。
- 初始聚类中心的选择会影响算法的收敛速度和最终结果。
- 对噪声和异常值比较敏感。

### 区别

- **学习类型**：KNN是监督学习，需要标注的训练数据；而K-Means是非监督学习，不需要标注的数据。
- **应用场景**：KNN用于分类和回归，K-Means仅用于聚类。
- **数据要求**：KNN在预测时计算量大，K-Means在训练时计算量大。
- **算法复杂度**：KNN简单直观，K-Means需要迭代寻找聚类中心。

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




