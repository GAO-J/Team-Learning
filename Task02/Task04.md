# 异常检测——基于相似度的方法

**参考：**

https://github.com/datawhalechina/team-learning-data-mining/tree/master/AnomalyDetection

https://www.cnblogs.com/wj-1314/p/14049195.html

**知识图谱：**

<img src="D:\DataWhale\OutlierAnalysis\Task02\img\QQ图片20210520171922.jpg" alt="png" style="zoom: 50%;" />

**主要内容包括：**

- 基于距离的度量
- 基于密度的度量

[TOC]



## 1、概述

&emsp;&emsp;“异常”通常是一个主观的判断，什么样的数据被认为是“异常”的，需要结合业务背景和环境来具体分析确定。 
&emsp;&emsp;实际上，数据通常嵌入在大量的噪声中，而我们所说的“异常值”通常指具有特定业务意义的那一类特殊的异常值。噪声可以视作特性较弱的异常值，没有被分析的价值。噪声和异常之间、正常数据和噪声之间的边界都是模糊的。异常值通常具有更高的离群程度分数值，同时也更具有可解释性。    

&emsp;&emsp;在普通的数据处理中，我们常常需要保留正常数据，而对噪声和异常值的特性则基本忽略。但在异常检测中，我们弱化了“噪声”和“正常数据”之间的区别，专注于那些具有有价值特性的异常值。在基于相似度的方法中，主要思想是异常点的表示与正常点不同



## 2、基于距离的度量 

&emsp;&emsp;基于距离的方法是一种常见的适用于各种数据域的异常检测算法，它基于最近邻距离来定义异常值。 此类方法不仅适用于多维数值数据，在其他许多领域，例如分类数据，文本数据，时间序列数据和序列数据等方面也有广泛的应用。
&emsp;&emsp;基于距离的异常检测有这样一个前提假设，即异常点的 $k$ 近邻距离要远大于正常点。解决问题的最简单方法是使用嵌套循环。 第一层循环遍历每个数据，第二层循环进行异常判断，需要计算当前点与其他点的距离，一旦已识别出多于 $k$ 个数据点与当前点的距离在 $D$ 之内，则将该点自动标记为非异常值。 这样计算的时间复杂度为$O(N^{2})$，当数据量比较大时，这样计算是及不划算的。 因此，需要修剪方法以加快距离计算。

### 2.1 基于单元的方法

&emsp;&emsp;在基于单元格的技术中，数据空间被划分为单元格，单元格的宽度是阈值D和数据维数的函数。具体地说，每个维度被划分成宽度最多为 $\frac{D}{{2 \cdot \sqrt d }}$ 单元格。在给定的单元以及相邻的单元中存在的数据点满足某些特性，这些特性可以让数据被更有效的处理。

![@图 1 基于单元的数据空间分区 | center| 500x0](D:\DataWhale\team-learning-data-mining-master\AnomalyDetection\img\UWiX5C7kCHx5yX7O9yQm9F1ndg-QgMqS3BAwIWPB40k.original.fullsize-1609839833441.png)

&emsp;&emsp;以二维情况为例，此时网格间的距离为 $\frac{D}{{2 \cdot \sqrt d }}$ ，需要记住的一点是，网格单元的数量基于数据空间的分区，并且与数据点的数量无关。这是决定该方法在低维数据上的效率的重要因素，在这种情况下，网格单元的数量可能不多。 另一方面，此方法不适用于更高维度的数据。对于给定的单元格，其 $L_{1}$ 邻居被定义为通过最多1个单元间的边界可从该单元到达的单元格的集合。请注意，在一个角上接触的两个单元格也是 $L_{1}$ 邻居。  $L_{2}$ 邻居是通过跨越2个或3个边界而获得的那些单元格。 上图中显示了标记为 $X$的特定单元格及其 $L_{1}$ 和 $L_{2}$ 邻居集。 显然，内部单元具有8个 $L_{1}$ 邻居和40个 $L_{2}$ 邻居。 然后，可以立即观察到以下性质：

- 单元格中两点之间的距离最多为 $D/2$。
- 一个点与 $L_{1}$ 邻接点之间的距离最大为 $D$。
- 一个点与它的 $Lr$ 邻居(其中$r$ > 2)中的一个点之间的距离至少为$D$。

&emsp;&emsp;唯一无法直接得出结论的是 $L_{2}$ 中的单元格。 这表示特定单元中数据点的不确定性区域。 对于这些情况，需要明确执行距离计算。 同时，可以定义许多规则，以便立即将部分数据点确定为异常值或非异常值。 规则如下：

-  如果一个单元格中包含超过 $k$ 个数据点及其  $L_{1}$ 邻居，那么这些数据点都不是异常值。
-  如果单元 $A$ 及其相邻 $L_{1}$ 和 $L_{2}$ 中包含少于 $k$ 个数据点，则单元A中的所有点都是异常值。

&emsp;&emsp;此过程的第一步是将部分数据点直接标记为非异常值（如果由于第一个规则而导致它们的单元格包含 $k$ 个点以上）。 此外，此类单元格的所有相邻单元格仅包含非异常值。 为了充分利用第一条规则的修剪能力，确定每个单元格及其 $L_{1}$ 邻居中点的总和。 如果总数大于 $k$ ，则所有这些点也都标记为非离群值。

&emsp;&emsp;接下来，利用第二条规则的修剪能力。 对于包含至少一个数据点的每个单元格 $A$，计算其中的点数及其   $L_{1}$ 和  $L_{2}$ 邻居的总和。 如果该数字不超过 $k$，则将单元格$A$ 中的所有点标记为离群值。 此时，许多单元可能被标记为异常值或非异常值。 

&emsp;&emsp;对于此时仍未标记为异常值或非异常值的单元格中的数据点需要明确计算其 $k$ 最近邻距离。即使对于这样的数据点，通过使用单元格结构也可以更快地计算出 $k$ 个最近邻的距离。考虑到目前为止尚未被标记为异常值或非异常值的单元格$A$。这样的单元可能同时包含异常值和非异常值。单元格 $A$ 中数据点的不确定性主要存在于该单元格的 $L_{2}$ 邻居中的点集。无法通过规则知道 $A$ 的 $L_{2}$ 邻居中的点是否在阈值距离 $D$ 内，为了确定单元 $A$ 中数据点与其$L_{2}$ 邻居中的点集在阈值距离 $D$ 内的点数，需要进行显式距离计算。对于那些在  $L_{1}$ 和  $L_{2}$ 中不超过 $k$ 个且距离小于 $D$ 的数据点，则声明为异常值。需要注意，仅需要对单元 $A$ 中的点到单元$A$的$L_{2}$邻居中的点执行显式距离计算。这是因为已知 $L_{1}$ 邻居中的所有点到 $A$ 中任何点的距离都小于 $D$，并且已知 $Lr$ 中 $(r> 2)$ 的所有点与 $A$上任何点的距离至少为 $D$。因此，可以在距离计算中实现额外的节省。

### 2.2 基于索引的方法

&emsp;&emsp;对于一个给定数据集，基于索引的方法利用多维索引结构(如 $\mathrm{R}$ 树、$k-d$ 树)来搜索每个数据对象 $A$ 在半径 $D$ 范围 内的相邻点。设 $M$ 是一个异常值在其 $D$ -邻域内允许含有对象的最多个数，若发现某个数据对象 $A$ 的 $D$ -邻域内出现 $M+1$ 甚至更多个相邻点， 则判定对象 $A$ 不是异常值。该算法时间复杂度在最坏情况下为 $O\left(k N^{2}\right),$ 其中 $k$ 是数据集维数， $N$ 是数据集包含对象的个数。该算法在数据集的维数增加时具有较好的扩展性，但是时间复杂度的估算仅考虑了搜索时间，而构造索引的任务本身就需要密集复杂的计算量。

## 3、基于密度的度量          

&emsp;&emsp;基于密度的算法主要有局部离群因子(LocalOutlierFactor,LOF)，以及LOCI、CLOF等基于LOF的改进算法。下面我们以LOF为例来进行详细的介绍和实践。

&emsp;&emsp;基于距离的检测适用于各个集群的密度较为均匀的情况。在下图中，离群点B容易被检出，而若要检测出较为接近集群的离群点A，则可能会将一些集群边缘的点当作离群点丢弃。而LOF等基于密度的算法则可以较好地适应密度不同的集群情况。   

![图4.1距离检测的困境-离群点A.png](/home/leungkafai/AnomalyDetection/img/图4.1距离检测的困境-离群点A-1609839836032.png)

&emsp;&emsp; 那么，这个基于密度的度量值是怎么得来的呢？还是要从距离的计算开始。类似k近邻的思路，首先我们也需要来定义一个“k-距离”。

### 3.1 k-距离（k-distance(p)）：    

&emsp;&emsp;对于数据集$D$中的给定对象$p$，对象$p$与数据集$D$中任意点$o$的距离为$d(p,o)$。我们把数据集$D$中与对象$p$距离最近的$k$个相邻点的最远距离表示为$k-distance(p)$，把距离对象$p$距离第$k$近的点表示为$o_k$，那么给定对象$p$和点$o_k$之间的距离$d(p,o_k)=k − d i s t a n c e ( p )$，满足：    

+ 在集合$D$中至少有不包括$p$在内的$k$个点 $o'$，其中$o'∈D\{p\}$，满足$d(p,o')≤d(p,o_k)$
+ 在集合$D$中最多有不包括$p$在内的$k-1$个点$o'$，其中$o'∈D\{p\}$，满足$d(p,o')<d(p,o_k)$    

&emsp;&emsp;直观一些理解，就是以对象$p$为中心，对数据集$D$中的所有点到$p$的距离进行排序，距离对象$p$第$k$近的点$o_k$与$p$之间的距离就是k-距离。

### 3.2 k-邻域（k-distance neighborhood）：    

&emsp;&emsp;由k-距离，我们扩展到一个点的集合——到对象$p$的距离小于等于k-距离的所有点的集合，我们称之为k-邻域：$N_{k − d i s t a n c e ( p )}( p ) = \{ q ∈ D \backslash\{ p \} ∣ d ( p , q ) ≤ k − d i s t a n c e ( p )\}
$。

+ k-邻域包含对象$p$的第$k$距离以内的所有点，包括第$k$距离点。

+ 对象$p$的第$k$邻域点的个数$ ∣ N_k(p)∣ ≥ k$。

&emsp;&emsp;在二维平面上展示出来的话，对象$p$的k-邻域实际上就是以对象$p$为圆心、k-距离为半径围成的圆形区域。就是说，k-邻域已经从“距离”这个概念延伸到“空间”了。

![img](https://www.hualigs.cn/image/608950de59262.jpg)

### 3.3 可达距离（reachability distance）：    

&emsp;&emsp;有了邻域的概念，我们可以按照到对象$o$的距离远近，将数据集$D$内的点按照到$o$的距离分为两类：    

+ 若$p_i$在对象$o$的k-邻域内，则可达距离就是给定点$p_i$关于对象o的k-距离；
+ 若$p_i$在对象$o$的k-邻域外，则可达距离就是给定点$p_i$关于对象o的实际距离。  

&emsp;&emsp;给定点$p_i$关于对象$o$的可达距离用数学公式可以表示为：

&emsp;&emsp;$$r e a c h−d i s t_ k ( p , o ) = m a x \{k−distance( o ) , d ( p , o )\}$$ 。    
&emsp;&emsp;这样的分类处理可以简化后续的计算，同时让得到的数值区分度更高。

![可达距离.jpg](https://www.hualigs.cn/image/608951ddb06ae.jpg)

&emsp;&emsp;如图：

+ $p_1$在对象$o$的k-邻域内，$d ( p_1 , o )<k−distance( o )$，

  可达距离$r e a c h−d i s t_ k ( p_1 , o ) = k−distance( o )$ ;

+ $p_2$在对象$o$的k-邻域外，$d ( p_2 , o )>k−distance( o )$，

  可达距离$r e a c h−d i s t_ k ( p_2 , o ) = d ( p_2 , o )$ ;

&emsp;&emsp;注意：这里用的是$p_k$与$o$的距离$d(p_k,o)$与$o$的k-距离$k−distance( o )$来进行比较，不是与$k−distance( p )$进行比较！

&emsp;&emsp;可达距离的设计是为了减少距离的计算开销，$o$的k-邻域内的所有对象$p$的k-距离计算量可以被显著降低，相当于使用一个阈值把需要计算的部分“截断”了。这种“截断”对计算量的降低效果可以通过参数$k$来控制，$k$的值越高，无需计算的邻近点越多，计算开销越小。但是另一方面，$k$的值变高，可能意味着可达距离变远，对集群点和离群点的区分度可能变低。因此，如何选择$k$值，是LOF算法能否达到效率与效果平衡的重要因素。

### 3.4 局部可达密度（local reachability density）：

&emsp;&emsp;我们可以将“密度”直观地理解为点的聚集程度，就是说，点与点之间距离越短，则密度越大。在这里，我们使用数据集$D$中对象$p$与对象$o$的k-邻域内所有点的可达距离平均值的倒数（注意，不是导数）来定义局部可达密度。  

&emsp;&emsp;在进行局部可达密度的计算的时候，我们需要避免数据集内所有数据落在同一点上，即所有可达距离之和为0的情况：此时局部密度为∞，后续计算将无法进行。LOF算法中针对这一问题进行了如下的定义：对于数据集$D$内的给定对象$p$，存在至少$MinPts(p)\geq1$个不同于$p$的点。因此，我们使用对象$p$到$o∈N_{MinPts}(p)$的可达距离$reach-dist_{MinPts}(p, o)$作为度量对象$p$邻域的密度的值。

&emsp;&emsp;给定点p的局部可达密度计算公式为：$$lrd_{MinPts}(p)=1/(\frac {\sum\limits_{o∈N_{MinPts}(p)} reach-dist_{MinPts}(p,o)} {\left\vert N_{MinPts}(p) \right\vert})$$ 

&emsp;&emsp;由公式可以看出，这里是对给定点p进行度量，计算其邻域内的所有对象o到给定点p的可达距离平均值。给定点p的局部可达密度越高，越可能与其邻域内的点	属于同一簇；密度越低，越可能是离群点。  

### 3.5 局部异常因子：    

&emsp;&emsp;得到lrd（局部可达密度）以后就可以将每个点的lrd将与它们的k个邻点的lrd进行比较，得到局部异常因子LOF。更具体地说，LOF在数学上是对象$p$的邻居点$o$（$o∈N_{MinPts}(p)$）的lrd平均值与$p$的lrd的比值。

![局部异常因子公式.png](https://www.hualigs.cn/image/6089531d9e0f7.jpg)

&emsp;&emsp;不难看出，$p$的局部可达密度越低，且它的$MinPts$近邻的平均局部可达密度越高，则$p$的LOF值越高。

&emsp;&emsp;如果这个比值越接近1，说明o的邻域点密度差不多，o可能和邻域同属一簇；如果这个比值小于1，说明o的密度高于其邻域点密度，o为密集点；如果这个比值大于1，说明o的密度小于其邻域点密度，o可能是异常点。

&emsp;&emsp;由公式计算出的LOF数值，就是我们所需要的离群点分数。



##  4、实战

### 4.1 LOF示例代码


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor as LOF
```


```python
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```


```python
np.random.seed(61)

# 构造两个数据点集群(正态分布的)
X_inliers1 = 0.2 * np.random.randn(100, 2)
X_inliers2 = 0.5 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers1 + 2, X_inliers2 - 2]

# 构造一些离群的点
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# 拼成训练集
X = np.r_[X_inliers, X_outliers]

n_outliers = len(X_outliers)
ground_truth = np.ones(len(X), dtype=int)
# 打标签，群内点构造离群值为1，离群点构造离群值为-1
ground_truth[-n_outliers:] = -1
```


```python
plt.title('构造数据集 (LOF)')
plt.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], color='b', s=5, label='集群点')
plt.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], color='orange', s=5, label='离群点')

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
legend = plt.legend(loc='upper left')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()
```


![png](img/Task04_5_0.png)
    



```python
# 训练模型（找出每个数据的实际离群值）
clf = LOF(n_neighbors=20, contamination=0.1)

# 对单个数据集进行无监督检测时，以1和-1分别表示非离群点与离群点
y_pred = clf.fit_predict(X)

# 找出构造离群值与实际离群值不同的点
n_errors = y_pred != ground_truth
X_pred = np.c_[X,n_errors]

X_scores = clf.negative_outlier_factor_
# 实际离群值有正有负，转化为正数并保留其差异性（不是直接取绝对值）
X_scores_nor = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
X_pred = np.c_[X_pred,X_scores_nor]
X_pred = pd.DataFrame(X_pred,columns=['x','y','pred','scores'])

X_pred_same = X_pred[X_pred['pred'] == False]
X_pred_different = X_pred[X_pred['pred'] == True]

# 直观地看一看数据
X_pred
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>pred</th>
      <th>scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.913701</td>
      <td>2.087875</td>
      <td>0.0</td>
      <td>0.000494</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.999748</td>
      <td>2.212225</td>
      <td>0.0</td>
      <td>0.005255</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.040673</td>
      <td>2.133115</td>
      <td>0.0</td>
      <td>0.001521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.791277</td>
      <td>1.743218</td>
      <td>0.0</td>
      <td>0.015652</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.991693</td>
      <td>1.770405</td>
      <td>0.0</td>
      <td>0.010113</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.974831</td>
      <td>2.204030</td>
      <td>0.0</td>
      <td>0.005188</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2.205924</td>
      <td>1.986534</td>
      <td>0.0</td>
      <td>0.022802</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.880050</td>
      <td>2.048036</td>
      <td>0.0</td>
      <td>0.002938</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.558993</td>
      <td>1.886954</td>
      <td>0.0</td>
      <td>0.044276</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.308499</td>
      <td>2.136235</td>
      <td>0.0</td>
      <td>0.034563</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2.025291</td>
      <td>1.723945</td>
      <td>0.0</td>
      <td>0.014572</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.851987</td>
      <td>1.914373</td>
      <td>0.0</td>
      <td>0.001505</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2.023122</td>
      <td>2.100367</td>
      <td>0.0</td>
      <td>0.002352</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.982056</td>
      <td>2.174584</td>
      <td>0.0</td>
      <td>0.001103</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.876272</td>
      <td>1.607466</td>
      <td>0.0</td>
      <td>0.027902</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2.290145</td>
      <td>1.561354</td>
      <td>1.0</td>
      <td>0.069235</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1.831982</td>
      <td>2.202031</td>
      <td>0.0</td>
      <td>0.012232</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.876901</td>
      <td>1.842190</td>
      <td>0.0</td>
      <td>0.004201</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1.930508</td>
      <td>2.344936</td>
      <td>0.0</td>
      <td>0.020550</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2.091724</td>
      <td>1.788213</td>
      <td>0.0</td>
      <td>0.020776</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.824822</td>
      <td>1.817189</td>
      <td>0.0</td>
      <td>0.005238</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1.590550</td>
      <td>1.963813</td>
      <td>0.0</td>
      <td>0.036123</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1.842309</td>
      <td>1.950124</td>
      <td>0.0</td>
      <td>0.003503</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1.921702</td>
      <td>1.822401</td>
      <td>0.0</td>
      <td>0.006309</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2.160526</td>
      <td>2.368397</td>
      <td>0.0</td>
      <td>0.045859</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1.918054</td>
      <td>2.337582</td>
      <td>0.0</td>
      <td>0.020388</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1.673496</td>
      <td>1.972128</td>
      <td>0.0</td>
      <td>0.017914</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2.010597</td>
      <td>1.988018</td>
      <td>0.0</td>
      <td>0.002593</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1.747966</td>
      <td>2.255164</td>
      <td>0.0</td>
      <td>0.026164</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2.099874</td>
      <td>2.109433</td>
      <td>0.0</td>
      <td>0.004094</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1.476653</td>
      <td>1.800120</td>
      <td>1.0</td>
      <td>0.071098</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1.768986</td>
      <td>2.113845</td>
      <td>0.0</td>
      <td>0.012098</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2.128567</td>
      <td>2.049832</td>
      <td>0.0</td>
      <td>0.007022</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1.682819</td>
      <td>2.286214</td>
      <td>0.0</td>
      <td>0.038134</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1.894061</td>
      <td>2.227710</td>
      <td>0.0</td>
      <td>0.012928</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2.064048</td>
      <td>2.138074</td>
      <td>0.0</td>
      <td>0.002496</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2.155524</td>
      <td>2.044044</td>
      <td>0.0</td>
      <td>0.010759</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1.800120</td>
      <td>2.067040</td>
      <td>0.0</td>
      <td>0.008211</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2.026954</td>
      <td>2.178671</td>
      <td>0.0</td>
      <td>0.001721</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1.741088</td>
      <td>2.013464</td>
      <td>0.0</td>
      <td>0.010708</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1.890347</td>
      <td>2.366134</td>
      <td>0.0</td>
      <td>0.025904</td>
    </tr>
    <tr>
      <th>41</th>
      <td>1.964812</td>
      <td>1.695022</td>
      <td>0.0</td>
      <td>0.011949</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1.952727</td>
      <td>2.197472</td>
      <td>0.0</td>
      <td>0.005465</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2.155542</td>
      <td>1.921827</td>
      <td>0.0</td>
      <td>0.020703</td>
    </tr>
    <tr>
      <th>44</th>
      <td>1.872897</td>
      <td>1.779439</td>
      <td>0.0</td>
      <td>0.008031</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2.117473</td>
      <td>2.124645</td>
      <td>0.0</td>
      <td>0.005732</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2.162764</td>
      <td>1.505490</td>
      <td>0.0</td>
      <td>0.068432</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1.988108</td>
      <td>2.036029</td>
      <td>0.0</td>
      <td>0.000536</td>
    </tr>
    <tr>
      <th>48</th>
      <td>2.371013</td>
      <td>2.080772</td>
      <td>0.0</td>
      <td>0.048905</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1.983282</td>
      <td>2.076261</td>
      <td>0.0</td>
      <td>0.001085</td>
    </tr>
    <tr>
      <th>50</th>
      <td>2.080582</td>
      <td>2.361234</td>
      <td>0.0</td>
      <td>0.030681</td>
    </tr>
    <tr>
      <th>51</th>
      <td>1.847340</td>
      <td>1.423799</td>
      <td>1.0</td>
      <td>0.081568</td>
    </tr>
    <tr>
      <th>52</th>
      <td>1.779722</td>
      <td>2.132461</td>
      <td>0.0</td>
      <td>0.011331</td>
    </tr>
    <tr>
      <th>53</th>
      <td>2.016613</td>
      <td>1.981629</td>
      <td>0.0</td>
      <td>0.004023</td>
    </tr>
    <tr>
      <th>54</th>
      <td>2.077056</td>
      <td>2.131391</td>
      <td>0.0</td>
      <td>0.002932</td>
    </tr>
    <tr>
      <th>55</th>
      <td>2.003966</td>
      <td>2.071045</td>
      <td>0.0</td>
      <td>0.002202</td>
    </tr>
    <tr>
      <th>56</th>
      <td>1.755985</td>
      <td>2.014354</td>
      <td>0.0</td>
      <td>0.009262</td>
    </tr>
    <tr>
      <th>57</th>
      <td>1.819063</td>
      <td>1.798662</td>
      <td>0.0</td>
      <td>0.008254</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2.113737</td>
      <td>1.596898</td>
      <td>0.0</td>
      <td>0.045737</td>
    </tr>
    <tr>
      <th>59</th>
      <td>1.907802</td>
      <td>2.305977</td>
      <td>0.0</td>
      <td>0.018199</td>
    </tr>
    <tr>
      <th>60</th>
      <td>1.975595</td>
      <td>1.977545</td>
      <td>0.0</td>
      <td>0.000513</td>
    </tr>
    <tr>
      <th>61</th>
      <td>2.154757</td>
      <td>2.143713</td>
      <td>0.0</td>
      <td>0.012042</td>
    </tr>
    <tr>
      <th>62</th>
      <td>1.788315</td>
      <td>1.869702</td>
      <td>0.0</td>
      <td>0.004738</td>
    </tr>
    <tr>
      <th>63</th>
      <td>1.823602</td>
      <td>2.042652</td>
      <td>0.0</td>
      <td>0.006435</td>
    </tr>
    <tr>
      <th>64</th>
      <td>2.096128</td>
      <td>1.903903</td>
      <td>0.0</td>
      <td>0.012023</td>
    </tr>
    <tr>
      <th>65</th>
      <td>1.923008</td>
      <td>1.850860</td>
      <td>0.0</td>
      <td>0.002889</td>
    </tr>
    <tr>
      <th>66</th>
      <td>1.873306</td>
      <td>2.046155</td>
      <td>0.0</td>
      <td>0.002727</td>
    </tr>
    <tr>
      <th>67</th>
      <td>1.940945</td>
      <td>2.319470</td>
      <td>0.0</td>
      <td>0.016756</td>
    </tr>
    <tr>
      <th>68</th>
      <td>2.147756</td>
      <td>2.236818</td>
      <td>0.0</td>
      <td>0.017172</td>
    </tr>
    <tr>
      <th>69</th>
      <td>2.393796</td>
      <td>1.874065</td>
      <td>0.0</td>
      <td>0.057395</td>
    </tr>
    <tr>
      <th>70</th>
      <td>2.363186</td>
      <td>2.262029</td>
      <td>0.0</td>
      <td>0.059440</td>
    </tr>
    <tr>
      <th>71</th>
      <td>2.059799</td>
      <td>2.062782</td>
      <td>0.0</td>
      <td>0.003571</td>
    </tr>
    <tr>
      <th>72</th>
      <td>2.301837</td>
      <td>2.124948</td>
      <td>0.0</td>
      <td>0.034683</td>
    </tr>
    <tr>
      <th>73</th>
      <td>2.180689</td>
      <td>1.812641</td>
      <td>0.0</td>
      <td>0.027560</td>
    </tr>
    <tr>
      <th>74</th>
      <td>1.887507</td>
      <td>2.272837</td>
      <td>0.0</td>
      <td>0.016182</td>
    </tr>
    <tr>
      <th>75</th>
      <td>1.658125</td>
      <td>1.776932</td>
      <td>0.0</td>
      <td>0.030200</td>
    </tr>
    <tr>
      <th>76</th>
      <td>1.898466</td>
      <td>1.888822</td>
      <td>0.0</td>
      <td>0.001174</td>
    </tr>
    <tr>
      <th>77</th>
      <td>1.523371</td>
      <td>1.933112</td>
      <td>0.0</td>
      <td>0.053989</td>
    </tr>
    <tr>
      <th>78</th>
      <td>1.971560</td>
      <td>2.084852</td>
      <td>0.0</td>
      <td>0.000245</td>
    </tr>
    <tr>
      <th>79</th>
      <td>1.853063</td>
      <td>1.948827</td>
      <td>0.0</td>
      <td>0.003545</td>
    </tr>
    <tr>
      <th>80</th>
      <td>1.940607</td>
      <td>2.204286</td>
      <td>0.0</td>
      <td>0.008379</td>
    </tr>
    <tr>
      <th>81</th>
      <td>1.861199</td>
      <td>1.812638</td>
      <td>0.0</td>
      <td>0.005640</td>
    </tr>
    <tr>
      <th>82</th>
      <td>1.798244</td>
      <td>1.987648</td>
      <td>0.0</td>
      <td>0.005989</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2.095400</td>
      <td>1.774871</td>
      <td>0.0</td>
      <td>0.022915</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2.399160</td>
      <td>1.974885</td>
      <td>0.0</td>
      <td>0.056701</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2.106253</td>
      <td>2.140415</td>
      <td>0.0</td>
      <td>0.004863</td>
    </tr>
    <tr>
      <th>86</th>
      <td>1.845121</td>
      <td>2.016970</td>
      <td>0.0</td>
      <td>0.005105</td>
    </tr>
    <tr>
      <th>87</th>
      <td>1.982983</td>
      <td>1.873035</td>
      <td>0.0</td>
      <td>0.005797</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2.013177</td>
      <td>1.765327</td>
      <td>0.0</td>
      <td>0.010917</td>
    </tr>
    <tr>
      <th>89</th>
      <td>1.776870</td>
      <td>1.964626</td>
      <td>0.0</td>
      <td>0.005442</td>
    </tr>
    <tr>
      <th>90</th>
      <td>1.952865</td>
      <td>2.156336</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>91</th>
      <td>1.719132</td>
      <td>1.645017</td>
      <td>0.0</td>
      <td>0.041172</td>
    </tr>
    <tr>
      <th>92</th>
      <td>1.929617</td>
      <td>1.994502</td>
      <td>0.0</td>
      <td>0.000510</td>
    </tr>
    <tr>
      <th>93</th>
      <td>1.952448</td>
      <td>1.966636</td>
      <td>0.0</td>
      <td>0.000407</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2.213147</td>
      <td>1.998481</td>
      <td>0.0</td>
      <td>0.023092</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1.726845</td>
      <td>2.025845</td>
      <td>0.0</td>
      <td>0.012427</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2.040109</td>
      <td>1.978483</td>
      <td>0.0</td>
      <td>0.005102</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2.379007</td>
      <td>2.277010</td>
      <td>0.0</td>
      <td>0.060501</td>
    </tr>
    <tr>
      <th>98</th>
      <td>1.941922</td>
      <td>1.842525</td>
      <td>0.0</td>
      <td>0.005738</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1.885584</td>
      <td>1.841023</td>
      <td>0.0</td>
      <td>0.004809</td>
    </tr>
    <tr>
      <th>100</th>
      <td>-2.850371</td>
      <td>-1.864485</td>
      <td>0.0</td>
      <td>0.015768</td>
    </tr>
    <tr>
      <th>101</th>
      <td>-2.595049</td>
      <td>-1.869757</td>
      <td>0.0</td>
      <td>0.006324</td>
    </tr>
    <tr>
      <th>102</th>
      <td>-2.431365</td>
      <td>-2.317736</td>
      <td>0.0</td>
      <td>0.005888</td>
    </tr>
    <tr>
      <th>103</th>
      <td>-2.716496</td>
      <td>-2.494207</td>
      <td>0.0</td>
      <td>0.010056</td>
    </tr>
    <tr>
      <th>104</th>
      <td>-1.837394</td>
      <td>-2.152249</td>
      <td>0.0</td>
      <td>0.005972</td>
    </tr>
    <tr>
      <th>105</th>
      <td>-1.698166</td>
      <td>-0.625910</td>
      <td>1.0</td>
      <td>0.070995</td>
    </tr>
    <tr>
      <th>106</th>
      <td>-2.640491</td>
      <td>-1.626802</td>
      <td>0.0</td>
      <td>0.008366</td>
    </tr>
    <tr>
      <th>107</th>
      <td>-1.377627</td>
      <td>-2.729710</td>
      <td>0.0</td>
      <td>0.025818</td>
    </tr>
    <tr>
      <th>108</th>
      <td>-1.926686</td>
      <td>-2.036530</td>
      <td>0.0</td>
      <td>0.004315</td>
    </tr>
    <tr>
      <th>109</th>
      <td>-1.807869</td>
      <td>-1.098525</td>
      <td>0.0</td>
      <td>0.033388</td>
    </tr>
    <tr>
      <th>110</th>
      <td>-1.722510</td>
      <td>-2.195235</td>
      <td>0.0</td>
      <td>0.007064</td>
    </tr>
    <tr>
      <th>111</th>
      <td>-2.166980</td>
      <td>-2.373390</td>
      <td>0.0</td>
      <td>0.007679</td>
    </tr>
    <tr>
      <th>112</th>
      <td>-1.795051</td>
      <td>-1.239225</td>
      <td>0.0</td>
      <td>0.024723</td>
    </tr>
    <tr>
      <th>113</th>
      <td>-2.088587</td>
      <td>-1.498576</td>
      <td>0.0</td>
      <td>0.005728</td>
    </tr>
    <tr>
      <th>114</th>
      <td>-2.749856</td>
      <td>-2.098790</td>
      <td>0.0</td>
      <td>0.008870</td>
    </tr>
    <tr>
      <th>115</th>
      <td>-2.218066</td>
      <td>-1.953556</td>
      <td>0.0</td>
      <td>0.001296</td>
    </tr>
    <tr>
      <th>116</th>
      <td>-2.061078</td>
      <td>-1.882019</td>
      <td>0.0</td>
      <td>0.000990</td>
    </tr>
    <tr>
      <th>117</th>
      <td>-2.831248</td>
      <td>-1.835772</td>
      <td>0.0</td>
      <td>0.012744</td>
    </tr>
    <tr>
      <th>118</th>
      <td>-2.015682</td>
      <td>-1.432792</td>
      <td>0.0</td>
      <td>0.009616</td>
    </tr>
    <tr>
      <th>119</th>
      <td>-2.396110</td>
      <td>-1.614633</td>
      <td>0.0</td>
      <td>0.002973</td>
    </tr>
    <tr>
      <th>120</th>
      <td>-1.378249</td>
      <td>-1.803400</td>
      <td>0.0</td>
      <td>0.016099</td>
    </tr>
    <tr>
      <th>121</th>
      <td>-1.983460</td>
      <td>-1.765742</td>
      <td>0.0</td>
      <td>0.000889</td>
    </tr>
    <tr>
      <th>122</th>
      <td>-1.890638</td>
      <td>-1.326008</td>
      <td>0.0</td>
      <td>0.016250</td>
    </tr>
    <tr>
      <th>123</th>
      <td>-1.446198</td>
      <td>-1.958168</td>
      <td>0.0</td>
      <td>0.009149</td>
    </tr>
    <tr>
      <th>124</th>
      <td>-2.812372</td>
      <td>-2.447116</td>
      <td>0.0</td>
      <td>0.012302</td>
    </tr>
    <tr>
      <th>125</th>
      <td>-1.787338</td>
      <td>-1.967271</td>
      <td>0.0</td>
      <td>0.005335</td>
    </tr>
    <tr>
      <th>126</th>
      <td>-2.278931</td>
      <td>-3.515178</td>
      <td>0.0</td>
      <td>0.051574</td>
    </tr>
    <tr>
      <th>127</th>
      <td>-2.546236</td>
      <td>-0.937003</td>
      <td>0.0</td>
      <td>0.061929</td>
    </tr>
    <tr>
      <th>128</th>
      <td>-2.482133</td>
      <td>-1.580380</td>
      <td>0.0</td>
      <td>0.004464</td>
    </tr>
    <tr>
      <th>129</th>
      <td>-2.479536</td>
      <td>-2.915148</td>
      <td>0.0</td>
      <td>0.018253</td>
    </tr>
    <tr>
      <th>130</th>
      <td>-2.485952</td>
      <td>-1.575638</td>
      <td>0.0</td>
      <td>0.004620</td>
    </tr>
    <tr>
      <th>131</th>
      <td>-1.954336</td>
      <td>-1.862846</td>
      <td>0.0</td>
      <td>0.002373</td>
    </tr>
    <tr>
      <th>132</th>
      <td>-2.304165</td>
      <td>-1.921647</td>
      <td>0.0</td>
      <td>0.000578</td>
    </tr>
    <tr>
      <th>133</th>
      <td>-1.863717</td>
      <td>-2.023051</td>
      <td>0.0</td>
      <td>0.004440</td>
    </tr>
    <tr>
      <th>134</th>
      <td>-2.322007</td>
      <td>-2.877288</td>
      <td>0.0</td>
      <td>0.015073</td>
    </tr>
    <tr>
      <th>135</th>
      <td>-1.575830</td>
      <td>-1.833638</td>
      <td>0.0</td>
      <td>0.007760</td>
    </tr>
    <tr>
      <th>136</th>
      <td>-2.087098</td>
      <td>-2.824356</td>
      <td>0.0</td>
      <td>0.013795</td>
    </tr>
    <tr>
      <th>137</th>
      <td>-1.269507</td>
      <td>-2.662155</td>
      <td>0.0</td>
      <td>0.029151</td>
    </tr>
    <tr>
      <th>138</th>
      <td>-1.674171</td>
      <td>-1.845464</td>
      <td>0.0</td>
      <td>0.006290</td>
    </tr>
    <tr>
      <th>139</th>
      <td>-2.797840</td>
      <td>-2.021635</td>
      <td>0.0</td>
      <td>0.012580</td>
    </tr>
    <tr>
      <th>140</th>
      <td>-0.897295</td>
      <td>-2.089731</td>
      <td>0.0</td>
      <td>0.028265</td>
    </tr>
    <tr>
      <th>141</th>
      <td>-2.686278</td>
      <td>-2.229461</td>
      <td>0.0</td>
      <td>0.009152</td>
    </tr>
    <tr>
      <th>142</th>
      <td>-2.116413</td>
      <td>-1.717624</td>
      <td>0.0</td>
      <td>0.000697</td>
    </tr>
    <tr>
      <th>143</th>
      <td>-1.656410</td>
      <td>-2.343056</td>
      <td>0.0</td>
      <td>0.011567</td>
    </tr>
    <tr>
      <th>144</th>
      <td>-2.458348</td>
      <td>-1.688111</td>
      <td>0.0</td>
      <td>0.003242</td>
    </tr>
    <tr>
      <th>145</th>
      <td>-1.229618</td>
      <td>-2.803122</td>
      <td>0.0</td>
      <td>0.034705</td>
    </tr>
    <tr>
      <th>146</th>
      <td>-2.026407</td>
      <td>-1.481229</td>
      <td>0.0</td>
      <td>0.006869</td>
    </tr>
    <tr>
      <th>147</th>
      <td>-2.824912</td>
      <td>-2.097260</td>
      <td>0.0</td>
      <td>0.011272</td>
    </tr>
    <tr>
      <th>148</th>
      <td>-1.341187</td>
      <td>-1.750172</td>
      <td>0.0</td>
      <td>0.018509</td>
    </tr>
    <tr>
      <th>149</th>
      <td>-2.424705</td>
      <td>-1.465733</td>
      <td>0.0</td>
      <td>0.004553</td>
    </tr>
    <tr>
      <th>150</th>
      <td>-0.994908</td>
      <td>-1.549343</td>
      <td>0.0</td>
      <td>0.033002</td>
    </tr>
    <tr>
      <th>151</th>
      <td>-1.383308</td>
      <td>-1.791218</td>
      <td>0.0</td>
      <td>0.015988</td>
    </tr>
    <tr>
      <th>152</th>
      <td>-1.975055</td>
      <td>-2.954678</td>
      <td>0.0</td>
      <td>0.021803</td>
    </tr>
    <tr>
      <th>153</th>
      <td>-1.777153</td>
      <td>-1.276620</td>
      <td>0.0</td>
      <td>0.021633</td>
    </tr>
    <tr>
      <th>154</th>
      <td>-2.827895</td>
      <td>-2.581811</td>
      <td>0.0</td>
      <td>0.013191</td>
    </tr>
    <tr>
      <th>155</th>
      <td>-2.440314</td>
      <td>-1.968343</td>
      <td>0.0</td>
      <td>0.001572</td>
    </tr>
    <tr>
      <th>156</th>
      <td>-2.280212</td>
      <td>-1.751333</td>
      <td>0.0</td>
      <td>0.001796</td>
    </tr>
    <tr>
      <th>157</th>
      <td>-2.052055</td>
      <td>-2.589392</td>
      <td>0.0</td>
      <td>0.008307</td>
    </tr>
    <tr>
      <th>158</th>
      <td>-1.543891</td>
      <td>-2.069898</td>
      <td>0.0</td>
      <td>0.007883</td>
    </tr>
    <tr>
      <th>159</th>
      <td>-1.714670</td>
      <td>-3.584388</td>
      <td>0.0</td>
      <td>0.049628</td>
    </tr>
    <tr>
      <th>160</th>
      <td>-2.257159</td>
      <td>-2.636035</td>
      <td>0.0</td>
      <td>0.010375</td>
    </tr>
    <tr>
      <th>161</th>
      <td>-2.176604</td>
      <td>-1.239460</td>
      <td>0.0</td>
      <td>0.020202</td>
    </tr>
    <tr>
      <th>162</th>
      <td>-1.846733</td>
      <td>-1.908060</td>
      <td>0.0</td>
      <td>0.004399</td>
    </tr>
    <tr>
      <th>163</th>
      <td>-2.405963</td>
      <td>-1.804772</td>
      <td>0.0</td>
      <td>0.002707</td>
    </tr>
    <tr>
      <th>164</th>
      <td>-1.152166</td>
      <td>-1.234453</td>
      <td>0.0</td>
      <td>0.031987</td>
    </tr>
    <tr>
      <th>165</th>
      <td>-2.701599</td>
      <td>-1.777598</td>
      <td>0.0</td>
      <td>0.007317</td>
    </tr>
    <tr>
      <th>166</th>
      <td>-1.596545</td>
      <td>-1.291331</td>
      <td>0.0</td>
      <td>0.016145</td>
    </tr>
    <tr>
      <th>167</th>
      <td>-2.363831</td>
      <td>-2.719461</td>
      <td>0.0</td>
      <td>0.010148</td>
    </tr>
    <tr>
      <th>168</th>
      <td>-1.766105</td>
      <td>-2.121741</td>
      <td>0.0</td>
      <td>0.005458</td>
    </tr>
    <tr>
      <th>169</th>
      <td>-2.024625</td>
      <td>-2.853125</td>
      <td>0.0</td>
      <td>0.015124</td>
    </tr>
    <tr>
      <th>170</th>
      <td>-2.079709</td>
      <td>-1.756540</td>
      <td>0.0</td>
      <td>0.000668</td>
    </tr>
    <tr>
      <th>171</th>
      <td>-1.662090</td>
      <td>-2.922772</td>
      <td>0.0</td>
      <td>0.021885</td>
    </tr>
    <tr>
      <th>172</th>
      <td>-2.403260</td>
      <td>-1.737711</td>
      <td>0.0</td>
      <td>0.002482</td>
    </tr>
    <tr>
      <th>173</th>
      <td>-2.130436</td>
      <td>-2.115826</td>
      <td>0.0</td>
      <td>0.003385</td>
    </tr>
    <tr>
      <th>174</th>
      <td>-1.528037</td>
      <td>-2.752665</td>
      <td>0.0</td>
      <td>0.022995</td>
    </tr>
    <tr>
      <th>175</th>
      <td>-2.274080</td>
      <td>-1.370711</td>
      <td>0.0</td>
      <td>0.007538</td>
    </tr>
    <tr>
      <th>176</th>
      <td>-2.463773</td>
      <td>-2.409219</td>
      <td>0.0</td>
      <td>0.007737</td>
    </tr>
    <tr>
      <th>177</th>
      <td>-1.640195</td>
      <td>-2.762050</td>
      <td>0.0</td>
      <td>0.020410</td>
    </tr>
    <tr>
      <th>178</th>
      <td>-2.355910</td>
      <td>-1.798011</td>
      <td>0.0</td>
      <td>0.002791</td>
    </tr>
    <tr>
      <th>179</th>
      <td>-2.011085</td>
      <td>-2.222525</td>
      <td>0.0</td>
      <td>0.004035</td>
    </tr>
    <tr>
      <th>180</th>
      <td>-2.149364</td>
      <td>-2.303450</td>
      <td>0.0</td>
      <td>0.005362</td>
    </tr>
    <tr>
      <th>181</th>
      <td>-1.290906</td>
      <td>-2.466642</td>
      <td>0.0</td>
      <td>0.022978</td>
    </tr>
    <tr>
      <th>182</th>
      <td>-2.236407</td>
      <td>-1.447560</td>
      <td>0.0</td>
      <td>0.004392</td>
    </tr>
    <tr>
      <th>183</th>
      <td>-2.189039</td>
      <td>-1.744755</td>
      <td>0.0</td>
      <td>0.001668</td>
    </tr>
    <tr>
      <th>184</th>
      <td>-2.800959</td>
      <td>-2.637256</td>
      <td>0.0</td>
      <td>0.014930</td>
    </tr>
    <tr>
      <th>185</th>
      <td>-0.705763</td>
      <td>-1.592191</td>
      <td>0.0</td>
      <td>0.052284</td>
    </tr>
    <tr>
      <th>186</th>
      <td>-2.384985</td>
      <td>-2.326095</td>
      <td>0.0</td>
      <td>0.004866</td>
    </tr>
    <tr>
      <th>187</th>
      <td>-1.858078</td>
      <td>-1.489428</td>
      <td>0.0</td>
      <td>0.008884</td>
    </tr>
    <tr>
      <th>188</th>
      <td>-3.168563</td>
      <td>-2.296149</td>
      <td>0.0</td>
      <td>0.029940</td>
    </tr>
    <tr>
      <th>189</th>
      <td>-1.597623</td>
      <td>-2.404847</td>
      <td>0.0</td>
      <td>0.015463</td>
    </tr>
    <tr>
      <th>190</th>
      <td>-0.841218</td>
      <td>-1.573119</td>
      <td>0.0</td>
      <td>0.039936</td>
    </tr>
    <tr>
      <th>191</th>
      <td>-2.359031</td>
      <td>-1.711905</td>
      <td>0.0</td>
      <td>0.002418</td>
    </tr>
    <tr>
      <th>192</th>
      <td>-1.636367</td>
      <td>-1.561806</td>
      <td>0.0</td>
      <td>0.009478</td>
    </tr>
    <tr>
      <th>193</th>
      <td>-2.378416</td>
      <td>-2.137452</td>
      <td>0.0</td>
      <td>0.002799</td>
    </tr>
    <tr>
      <th>194</th>
      <td>-2.654270</td>
      <td>-2.248436</td>
      <td>0.0</td>
      <td>0.008964</td>
    </tr>
    <tr>
      <th>195</th>
      <td>-2.653282</td>
      <td>-2.654251</td>
      <td>0.0</td>
      <td>0.011303</td>
    </tr>
    <tr>
      <th>196</th>
      <td>-0.924701</td>
      <td>-1.434561</td>
      <td>0.0</td>
      <td>0.038848</td>
    </tr>
    <tr>
      <th>197</th>
      <td>-2.096652</td>
      <td>-2.510361</td>
      <td>0.0</td>
      <td>0.007017</td>
    </tr>
    <tr>
      <th>198</th>
      <td>-2.624421</td>
      <td>-2.165664</td>
      <td>0.0</td>
      <td>0.007005</td>
    </tr>
    <tr>
      <th>199</th>
      <td>-1.810111</td>
      <td>-1.649138</td>
      <td>0.0</td>
      <td>0.005322</td>
    </tr>
    <tr>
      <th>200</th>
      <td>1.700910</td>
      <td>2.627236</td>
      <td>0.0</td>
      <td>0.106039</td>
    </tr>
    <tr>
      <th>201</th>
      <td>3.028379</td>
      <td>0.637051</td>
      <td>0.0</td>
      <td>0.423487</td>
    </tr>
    <tr>
      <th>202</th>
      <td>2.201209</td>
      <td>2.672844</td>
      <td>0.0</td>
      <td>0.115843</td>
    </tr>
    <tr>
      <th>203</th>
      <td>2.130066</td>
      <td>3.598384</td>
      <td>0.0</td>
      <td>0.394357</td>
    </tr>
    <tr>
      <th>204</th>
      <td>0.095457</td>
      <td>-2.755787</td>
      <td>0.0</td>
      <td>0.106312</td>
    </tr>
    <tr>
      <th>205</th>
      <td>-1.082280</td>
      <td>3.789883</td>
      <td>0.0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>206</th>
      <td>2.835815</td>
      <td>3.705321</td>
      <td>0.0</td>
      <td>0.477295</td>
    </tr>
    <tr>
      <th>207</th>
      <td>2.123381</td>
      <td>2.532351</td>
      <td>0.0</td>
      <td>0.069239</td>
    </tr>
    <tr>
      <th>208</th>
      <td>-2.758790</td>
      <td>0.508364</td>
      <td>0.0</td>
      <td>0.257763</td>
    </tr>
    <tr>
      <th>209</th>
      <td>-1.815366</td>
      <td>0.500731</td>
      <td>0.0</td>
      <td>0.208484</td>
    </tr>
    <tr>
      <th>210</th>
      <td>2.578535</td>
      <td>-0.991289</td>
      <td>0.0</td>
      <td>0.773669</td>
    </tr>
    <tr>
      <th>211</th>
      <td>3.208996</td>
      <td>1.290597</td>
      <td>0.0</td>
      <td>0.322943</td>
    </tr>
    <tr>
      <th>212</th>
      <td>-2.633439</td>
      <td>-3.628118</td>
      <td>0.0</td>
      <td>0.072436</td>
    </tr>
    <tr>
      <th>213</th>
      <td>1.940575</td>
      <td>-0.573875</td>
      <td>0.0</td>
      <td>0.669449</td>
    </tr>
    <tr>
      <th>214</th>
      <td>-2.112364</td>
      <td>1.186219</td>
      <td>0.0</td>
      <td>0.310505</td>
    </tr>
    <tr>
      <th>215</th>
      <td>1.260683</td>
      <td>1.964384</td>
      <td>0.0</td>
      <td>0.125260</td>
    </tr>
    <tr>
      <th>216</th>
      <td>0.092434</td>
      <td>-2.721035</td>
      <td>0.0</td>
      <td>0.108697</td>
    </tr>
    <tr>
      <th>217</th>
      <td>-3.148785</td>
      <td>3.659612</td>
      <td>0.0</td>
      <td>0.825292</td>
    </tr>
    <tr>
      <th>218</th>
      <td>-1.124219</td>
      <td>-1.593130</td>
      <td>1.0</td>
      <td>0.025813</td>
    </tr>
    <tr>
      <th>219</th>
      <td>-0.628741</td>
      <td>-1.696327</td>
      <td>1.0</td>
      <td>0.054695</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.title('局部离群因子检测 (LOF)')
plt.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], color='b', s=5, label='集群点')
plt.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], color='orange', s=5, label='离群点')

# 以标准化之后的局部离群值为半径画圆，以圆的大小直观表示出每个数据点的离群程度
plt.scatter(X_pred_same.values[:,0], X_pred_same.values[:, 1], 
            s=1000 * X_pred_same.values[:, 3], edgecolors='c', 
            facecolors='none', label='标签一致')
plt.scatter(X_pred_different.values[:, 0], X_pred_different.values[:, 1], 
            s=1000 * X_pred_different.values[:, 3], edgecolors='violet', 
            facecolors='none', label='标签不同')

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))

legend = plt.legend(loc='upper left')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()
```


![png](img/Task04_7_0.png)
    


### 4.2 实战二

在一组数中找异常点


```python
import numpy as np
from sklearn.neighbors import LocalOutlierFactor as LOF
 
X = [[-1.1], [0.2], [100.1], [0.3]]
clf = LOF(n_neighbors=2)
res = clf.fit_predict(X)
print(res)
print(clf.negative_outlier_factor_)
```

    [ 1  1 -1  1]
    [ -0.98214286  -1.03703704 -72.64219576  -0.98214286]


异常检测


```python
import numpy as np
from sklearn.neighbors import LocalOutlierFactor as LOF
import matplotlib.pyplot as plt
```


```python
#构造训练数据
X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]

#构造离群点
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

#拼成训练集
X = np.r_[X_inliers, X_outliers]

n_outliers = len(X_outliers)  # 20
ground_truth = np.ones(len(X), dtype=int)
ground_truth[-n_outliers:] = -1
```


```python
plt.title('构造数据集 (LOF)')
plt.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], color='b', s=5, label='集群点')
plt.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], color='orange', s=5, label='离群点')

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
legend = plt.legend(loc='upper left')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()
```


![png](img/Task04_14_0.png)
    



```python
# 训练模型（找出每个数据的实际离群值）
clf = LOF(n_neighbors=20, contamination=0.1)
 
# 对单个数据集进行无监督检测时，以1和-1分别表示非离群点与离群点
y_pred = clf.fit_predict(X)

# 找出构造离群值与实际离群值不同的点
n_errors = (y_pred != ground_truth).sum()
X_scores = clf.negative_outlier_factor_
```


```python
# 以标准化之后的局部离群值为半径画圆，以圆的大小直观表示出每个数据点的离群程度
plt.title('Locla Outlier Factor (LOF)')
plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')

radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
plt.scatter(X[:, 0], X[:, 1], s=1000*radius, edgecolors='r',
    facecolors='none', label='Outlier scores')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.xlabel("prediction errors: %d"%(n_errors))
legend = plt.legend(loc='upper left')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()
```


![png](img/Task04_16_0.png)
    


Novelty detection


```python
import numpy as np
from sklearn.neighbors import LocalOutlierFactor as LOF
import matplotlib.pyplot as plt
import matplotlib
```


```python
# np.meshgrid() 生成网格坐标点
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
 
# generate normal  (not abnormal) training observations 
X = 0.3*np.random.randn(100, 2)
X_train = np.r_[X+2, X-2]
 
# generate new normal (not abnormal) observations
X = 0.3*np.random.randn(20, 2)
X_test = np.r_[X+2, X-2]
 
# generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
 
# fit the model for novelty detection  (novelty=True)
clf = LOF(n_neighbors=20, contamination=0.1, novelty=True)
clf.fit(X_train)
 
# do not use predict, decision_function and score_samples on X_train
# as this would give wrong results but only on new unseen data(not
# used in X_train , eg: X_test, X_outliers or the meshgrid)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
```


```python
### contamination=0.01

# X_test: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
# y_pred_outliers: [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]

n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
 
# plot the learned frontier, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
 
plt.title('Novelty Detection with LOF')
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
 
s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s, edgecolors='k')
 
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s, edgecolors='k')
 
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
            ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
            loc='upper left',
            prop=matplotlib.font_manager.FontProperties(size=11))
 
plt.xlabel("errors novel regular:%d/40; errors novel abnormal: %d/40"
    %(n_error_test, n_error_outliers))
plt.show()
```


![png](img/Task04_20_0.png)
    

