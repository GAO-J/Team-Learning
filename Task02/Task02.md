主要参考：

https://github.com/datawhalechina/team-learning-data-mining/tree/master/AnomalyDetection

https://zhuanlan.zhihu.com/p/37753692

# 1、概述

统计学方法对数据的正常性做出假定。**它们假定正常的数据对象由一个统计模型产生，而不遵守该模型的数据是异常点。**统计学方法的有效性高度依赖于对给定数据所做的统计模型假定是否成立。

异常检测的统计学方法的一般思想是：学习一个拟合给定数据集的生成模型，然后识别该模型低概率区域中的对象，把它们作为异常点。

即利用统计学方法建立一个模型，然后考虑对象有多大可能符合该模型。

根据如何指定和学习模型，异常检测的统计学方法可以划分为两个主要类型：参数方法和非参数方法。

**参数方法**假定正常的数据对象被一个以$\Theta$为参数的参数分布产生。该参数分布的概率密度函数$f(x,\Theta)$给出对象$x$被该分布产生的概率。该值越小，$x$越可能是异常点。

**非参数方法**并不假定先验统计模型，而是试图从输入数据确定模型。非参数方法通常假定参数的个数和性质都是灵活的，不预先确定（所以非参数方法并不是说模型是完全无参的，完全无参的情况下从数据学习模型是不可能的）。

# 2、参数方法

## 2.1 基于正态分布的一元异常点检测

仅涉及一个属性或变量的数据称为一元数据。我们假定数据由正态分布产生，然后可以由输入数据学习正态分布的参数，并把低概率的点识别为异常点。

假定输入数据集为$\{x^{(1)}, x^{(2)}, ..., x^{(m)}\}$，数据集中的样本服从正态分布，即$x^{(i)}\sim N(\mu, \sigma^2)$，我们可以根据样本求出参数$\mu$和$\sigma$。

$\mu=\frac 1m\sum_{i=1}^m x^{(i)}$

$\sigma^2=\frac 1m\sum_{i=1}^m (x^{(i)}-\mu)^2$

求出参数之后，我们就可以根据概率密度函数计算数据点服从该分布的概率。正态分布的概率密度函数为

$p(x)=\frac 1{\sqrt{2\pi}\sigma}exp(-\frac{(x-\mu)^2}{2\sigma^2})$

如果计算出来的概率低于阈值，就可以认为该数据点为异常点。

阈值是个经验值，可以选择在验证集上使得评估指标值最大（也就是效果最好）的阈值取值作为最终阈值。

例如常用的3$\sigma$原则中，如果数据点超过范围$(\mu-3\sigma, \mu+3\sigma)$，那么这些点很有可能是异常点。

这个方法还可以用于可视化。箱线图对数据分布做了一个简单的统计可视化，利用数据集的上下四分位数（Q1和Q3）、中点等形成。异常点常被定义为小于Q1－1.5IQR或大于Q3+1.5IQR的那些数据。

用Python画一个简单的箱线图：


```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#具有标准正态分布
data = np.random.randn(50000) * 20 + 20
sns.boxplot(data=data)
```




    <AxesSubplot:>




    
![png](Task02_files/Task02_6_1.png)
    


## **2.2 多元异常点检测**

涉及两个或多个属性或变量的数据称为多元数据。许多一元异常点检测方法都可以扩充，用来处理多元数据。其核心思想是把多元异常点检测任务转换成一元异常点检测问题。例如基于正态分布的一元异常点检测扩充到多元情形时，可以求出每一维度的均值和标准差。对于第$j$维：

$\mu_j=\frac 1m\sum_{i=1}^m x_j^{(i)}$

$\sigma_j^2=\frac 1m\sum_{i=1}^m (x_j^{(i)}-\mu_j)^2$

计算概率时的概率密度函数为

$p(x)=\prod_{j=1}^n p(x_j;\mu_j,\sigma_j^2)=\prod_{j=1}^n\frac 1{\sqrt{2\pi}\sigma_j}exp(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2})$

这是在各个维度的特征之间相互独立的情况下。如果特征之间有相关性，就要用到多元高斯分布了。



## **2.3 多个特征相关，且符合多元高斯分布**

$\mu=\frac{1}{m}\sum^m_{i=1}x^{(i)}$

$\sum=\frac{1}{m}\sum^m_{i=1}(x^{(i)}-\mu)(x^{(i)}-\mu)^T$

$p(x)=\frac{1}{(2 \pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}} \exp \left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)$



ps:当多元高斯分布模型的协方差矩阵$\sum$为对角矩阵，且对角线上的元素为各自一元高斯分布模型的方差时，二者是等价的。

## **2.4 使用混合参数分布**

在许多情况下假定数据是由正态分布产生的。当实际数据很复杂时，这种假定过于简单，可以假定数据是被混合参数分布产生的。

# 3、非参数方法

在异常检测的非参数方法中，“正常数据”的模型从输入数据学习，而不是假定一个先验。通常，非参数方法对数据做较少假定，因而在更多情况下都可以使用。

**例子：使用直方图检测异常点。**

直方图是一种频繁使用的非参数统计模型，可以用来检测异常点。该过程包括如下两步：

步骤1：构造直方图。使用输入数据（训练数据）构造一个直方图。该直方图可以是一元的，或者多元的（如果输入数据是多维的）。

尽管非参数方法并不假定任何先验统计模型，但是通常确实要求用户提供参数，以便由数据学习。例如，用户必须指定直方图的类型（等宽的或等深的）和其他参数（直方图中的箱数或每个箱的大小等）。与参数方法不同，这些参数并不指定数据分布的类型。

步骤2：检测异常点。为了确定一个对象是否是异常点，可以对照直方图检查它。在最简单的方法中，如果该对象落入直方图的一个箱中，则该对象被看作正常的，否则被认为是异常点。

对于更复杂的方法，可以使用直方图赋予每个对象一个异常点得分。例如令对象的异常点得分为该对象落入的箱的容积的倒数。

使用直方图作为异常点检测的非参数模型的一个缺点是，很难选择一个合适的箱尺寸。一方面，如果箱尺寸太小，则许多正常对象都会落入空的或稀疏的箱中，因而被误识别为异常点。另一方面，如果箱尺寸太大，则异常点对象可能渗入某些频繁的箱中，因而“假扮”成正常的。

# 4、基于角度的方法

基于角度的方法的主要思想是：数据边界上的数据很可能将整个数据包围在一个较小的角度内，而内部的数据点则可能以不同的角度围绕着他们。如下图所示，其中点A是一个异常点，点B位于数据内部。

![image-20210506144705398](./img/image-20210506144705398.png)

如果数据点与其余点离得较远，则潜在角度可能越小。因此，具有较小角度谱的数据点是异常值，而具有较大角度谱的数据点不是异常值。

考虑三个点X,Y,Z。如果对于任意不同的点Y,Z，有：

$$
W \operatorname{Cos}(\overrightarrow{X Y}, \overrightarrow{X Z})=\frac{\langle\overline{X Y}, X Z\rangle}{\|X Y\|\|X Z\|}
$$
其中$||\space||$代表L2范数 , $< · > $代表点积。

这是一个加权余弦，因为分母包含L2-范数，其通过距离的逆加权进一步减小了异常点的加权角，这也对角谱产生了影响。然后，通过改变数据点Y和Z，保持X的值不变计算所有角度的方法。相应地，数据点X的基于角度的异常分数（ABOF）∈ D为：


$$
A B O F(X)=\operatorname{Var}_{\{Y, Z \in D\}} W \operatorname{Cos}(\overrightarrow{X Y}, \overrightarrow{X Z})
$$

# 5、HBOS


HBOS全名为：Histogram-based Outlier Score。它是一种单变量方法的组合，不能对特征之间的依赖关系进行建模，但是计算速度较快，对大数据集友好。其基本假设是数据集的每个维度相互独立。然后对每个维度进行区间(bin)划分，区间的密度越高，异常评分越低。
HBOS算法流程：

1.为每个数据维度做出数据直方图。对分类数据统计每个值的频数并计算相对频率。对数值数据根据分布的不同采用以下两种方法：

* 静态宽度直方图：标准的直方图构建方法，在值范围内使用k个等宽箱。样本落入每个桶的频率（相对数量）作为密度（箱子高度）的估计。时间复杂度：$O(n)$

* 2.动态宽度直方图：首先对所有值进行排序，然后固定数量的$\frac{N}{k}$个连续值装进一个箱里，其中N是总实例数，k是箱个数；直方图中的箱面积表示实例数。因为箱的宽度是由箱中第一个值和最后一个值决定的，所有箱的面积都一样，因此每一个箱的高度都是可计算的。这意味着跨度大的箱的高度低，即密度小，只有一种情况例外，超过k个数相等，此时允许在同一个箱里超过$\frac{N}{k}$值。

  时间复杂度：$O(n\times log(n))$



2.对每个维度都计算了一个独立的直方图，其中每个箱子的高度表示密度的估计。然后为了使得最大高度为1（确保了每个特征与异常值得分的权重相等），对直方图进行归一化处理。最后，每一个实例的HBOS值由以下公式计算：



$$
H B O S(p)=\sum_{i=0}^{d} \log \left(\frac{1}{\text {hist}_{i}(p)}\right)
$$

**推导过程**：

假设样本*p*第 *i* 个特征的概率密度为$p_i(p)$ ，则*p*的概率密度可以计算为：
$$
P(p)=P_{1}(p) P_{2}(p) \cdots P_{d}(p)
$$
两边取对数：
$$
\begin{aligned}
\log (P(p)) &=\log \left(P_{1}(p) P_{2}(p) \cdots P_{d}(p)\right) =\sum_{i=1}^{d} \log \left(P_{i}(p)\right)
\end{aligned}
$$
概率密度越大，异常评分越小，为了方便评分，两边乘以“-1”：
$$
-\log (P(p))=-1 \sum_{i=1}^{d} \log \left(P_{t}(p)\right)=\sum_{i=1}^{d} \frac{1}{\log \left(P_{i}(p)\right)}
$$
最后可得：
$$
H B O S(p)=-\log (P(p))=\sum_{i=1}^{d} \frac{1}{\log \left(P_{i}(p)\right)}
$$



## 5、总结



1.异常检测的统计学方法由数据学习模型，以区别正常的数据对象和异常点。使用统计学方法的一个优点是，异常检测可以是统计上无可非议的。当然，仅当对数据所做的统计假定满足实际约束时才为真。

2.HBOS在全局异常检测问题上表现良好，但不能检测局部异常值。但是HBOS比标准算法快得多，尤其是在大数据集上。

# 6、实例-LOF

LOF主要参数
n_neighbors（int，optional （default = 20 ）- 默认情况下用于kneighbors查询的邻居数。如果n_neighbors大于提供的样本数，则将使用所有样本。

algorithm（{‘auto’ ，‘ball_tree’ ，‘kd_tree’ ，‘brute’} ，可选） 

用于计算最近邻居的算法：
'ball_tree’将使用BallTree
'kd_tree’将使用KDTree
'brute’将使用蛮力搜索。
'auto’将尝试根据传递给fit()方法的值来确定最合适的算法。
注意：在稀疏输入上拟合将使用强力来覆盖此参数的设置。

leaf_size（int，optional （default = 30 ）） - 传递给BallTree或KDTree的叶子大小。
这可能会影响构造和查询的速度，以及存储树所需的内存。最佳值取决于问题的性质。

metric（字符串或可调用，默认’minkowski’） -用于距离计算的度量。
from scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
from scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]

p（整数，可选（默认值= 2 ）） - 来自sklearn.metrics.pairwise.pairwise_distances的Minkowski度量的参数。
当p = 1时，这相当于使用manhattan_distance（l1）。
对于p = 2使用euclidean_distance（l2）。
对于任意p，使用minkowski_distance（l_p)。


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
def localoutlierfactor(data, predict, k):
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(n_neighbors=k + 1, algorithm='auto', contamination=0.1, n_jobs=-1)
    clf.fit(data)
    # 记录 k 邻域距离
    predict['k distances'] = clf.kneighbors(predict)[0].max(axis=1)
    # 记录 LOF 离群因子，做相反数处理
    predict['local outlier factor'] = -clf._decision_function(predict.iloc[:, :-1])
    return predict

def plot_lof(result, method):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(8, 4)).add_subplot(111)
    plt.scatter(result[result['local outlier factor'] > method].index,
                result[result['local outlier factor'] > method]['local outlier factor'], c='red', s=50,
                marker='.', alpha=None,
                label='离群点')
    plt.scatter(result[result['local outlier factor'] <= method].index,
                result[result['local outlier factor'] <= method]['local outlier factor'], c='black', s=50,
                marker='.', alpha=None, label='正常点')
    plt.hlines(method, -2, 2 + max(result.index), linestyles='--')
    plt.xlim(-2, 2 + max(result.index))
    plt.title('LOF局部离群点检测', fontsize=13)
    plt.ylabel('局部离群因子', fontsize=15)
    plt.legend()
    plt.show()

def lof(data, predict=None, k=5, method=1, plot=False):
    import pandas as pd
    # 判断是否传入测试数据，若没有传入则测试数据赋值为训练数据
    try:
        if predict == None:
            predict = data.copy()
    except Exception:
        pass
    predict = pd.DataFrame(predict)
    # 计算 LOF 离群因子
    predict = localoutlierfactor(data, predict, k)
    if plot == True:
        plot_lof(predict, method)
    # 根据阈值划分离群点与正常点
    outliers = predict[predict['local outlier factor'] > method].sort_values(by='local outlier factor')
    inliers = predict[predict['local outlier factor'] <= method].sort_values(by='local outlier factor')
    return outliers, inliers
```


```python
posi = pd.read_excel(r'../data/已结束项目任务数据.xls')
lon = np.array(posi["任务gps经度"][:])  # 经度
lat = np.array(posi["任务gps 纬度"][:])  # 纬度
A = list(zip(lat, lon))  # 按照纬度-经度匹配

# 获取任务密度，取第5邻域，阈值为2（LOF大于2认为是离群值）
outliers1, inliers1 = lof(A, k=5, method=2)
```


```python
inliers1
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
      <th>0</th>
      <th>1</th>
      <th>k distances</th>
      <th>local outlier factor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>416</th>
      <td>22.953774</td>
      <td>113.406341</td>
      <td>0.014116</td>
      <td>-0.624968</td>
    </tr>
    <tr>
      <th>742</th>
      <td>22.941328</td>
      <td>113.676795</td>
      <td>0.021511</td>
      <td>-0.623104</td>
    </tr>
    <tr>
      <th>536</th>
      <td>22.941328</td>
      <td>113.676795</td>
      <td>0.021511</td>
      <td>-0.623104</td>
    </tr>
    <tr>
      <th>630</th>
      <td>22.922770</td>
      <td>113.668950</td>
      <td>0.021917</td>
      <td>-0.610707</td>
    </tr>
    <tr>
      <th>301</th>
      <td>23.550796</td>
      <td>113.596301</td>
      <td>0.005166</td>
      <td>-0.603643</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>410</th>
      <td>22.912291</td>
      <td>113.350764</td>
      <td>0.032169</td>
      <td>0.579587</td>
    </tr>
    <tr>
      <th>409</th>
      <td>22.839969</td>
      <td>113.349897</td>
      <td>0.072327</td>
      <td>0.593637</td>
    </tr>
    <tr>
      <th>294</th>
      <td>23.563411</td>
      <td>113.580385</td>
      <td>0.018121</td>
      <td>0.998628</td>
    </tr>
    <tr>
      <th>744</th>
      <td>23.221139</td>
      <td>112.924846</td>
      <td>0.058768</td>
      <td>1.180098</td>
    </tr>
    <tr>
      <th>224</th>
      <td>23.338871</td>
      <td>113.111065</td>
      <td>0.108733</td>
      <td>1.359963</td>
    </tr>
  </tbody>
</table>
<p>830 rows × 4 columns</p>
</div>




```python
outliers1
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
      <th>0</th>
      <th>1</th>
      <th>k distances</th>
      <th>local outlier factor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>425</th>
      <td>23.454574</td>
      <td>113.498304</td>
      <td>0.132545</td>
      <td>5.861116</td>
    </tr>
    <tr>
      <th>296</th>
      <td>23.878398</td>
      <td>113.539711</td>
      <td>0.324031</td>
      <td>14.444743</td>
    </tr>
    <tr>
      <th>295</th>
      <td>23.625476</td>
      <td>113.431396</td>
      <td>0.178774</td>
      <td>18.943854</td>
    </tr>
    <tr>
      <th>297</th>
      <td>23.723118</td>
      <td>113.739427</td>
      <td>0.222326</td>
      <td>27.284010</td>
    </tr>
    <tr>
      <th>302</th>
      <td>23.816108</td>
      <td>113.957929</td>
      <td>0.443798</td>
      <td>27.810731</td>
    </tr>
  </tbody>
</table>
</div>




```python
for k in [3,5,10]:
    plt.figure('k=%d'%k)
    outliers1, inliers1 = lof(A, k=k, method = 2)
    plt.scatter(np.array(A)[:,0],np.array(A)[:,1],s = 10,c='b',alpha = 0.5)
    plt.scatter(outliers1[0],outliers1[1],s = 10+outliers1['local outlier factor']*100,c='r',alpha = 0.2)
    plt.title('k=%d' % k)
```


    
![png](Task02_files/Task02_28_0.png)
    



    
![png](Task02_files/Task02_28_1.png)
    



    
![png](Task02_files/Task02_28_2.png)
    


由此可知，当K>5时,结果基本趋于稳定

Test2


```python
posi = pd.read_excel(r'../data/已结束项目任务数据.xls')
lon = np.array(posi["任务gps经度"][:])  # 经度
lat = np.array(posi["任务gps 纬度"][:])  # 纬度
A = list(zip(lat, lon))  # 按照纬度-经度匹配

posi = pd.read_excel(r'../data/会员信息数据.xls')
lon = np.array(posi["会员位置(GPS)经度"][:])  # 经度
lat = np.array(posi["会员位置(GPS)纬度"][:])  # 纬度
B = list(zip(lat, lon))  # 按照纬度-经度匹配

# 获取任务对会员密度，取第5邻域，阈值为2（LOF大于2认为是离群值）
outliers2, inliers2 = lof(B, A, k=5, method=2)
```


```python
outliers2
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
      <th>0</th>
      <th>1</th>
      <th>k distances</th>
      <th>local outlier factor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>297</th>
      <td>23.723118</td>
      <td>113.739427</td>
      <td>0.223941</td>
      <td>2.854522</td>
    </tr>
    <tr>
      <th>296</th>
      <td>23.878398</td>
      <td>113.539711</td>
      <td>0.327200</td>
      <td>3.064313</td>
    </tr>
    <tr>
      <th>384</th>
      <td>22.649463</td>
      <td>113.612985</td>
      <td>0.178139</td>
      <td>3.341646</td>
    </tr>
    <tr>
      <th>302</th>
      <td>23.816108</td>
      <td>113.957929</td>
      <td>0.445676</td>
      <td>7.500326</td>
    </tr>
  </tbody>
</table>
</div>




```python
inliers2
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
      <th>0</th>
      <th>1</th>
      <th>k distances</th>
      <th>local outlier factor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>143</th>
      <td>23.135442</td>
      <td>113.390008</td>
      <td>0.003329</td>
      <td>-0.532644</td>
    </tr>
    <tr>
      <th>482</th>
      <td>23.044908</td>
      <td>113.788434</td>
      <td>0.008245</td>
      <td>-0.494079</td>
    </tr>
    <tr>
      <th>713</th>
      <td>22.820653</td>
      <td>113.679324</td>
      <td>0.013851</td>
      <td>-0.468877</td>
    </tr>
    <tr>
      <th>740</th>
      <td>22.820653</td>
      <td>113.679324</td>
      <td>0.013851</td>
      <td>-0.468877</td>
    </tr>
    <tr>
      <th>746</th>
      <td>22.820653</td>
      <td>113.679324</td>
      <td>0.013851</td>
      <td>-0.468877</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>258</th>
      <td>23.014037</td>
      <td>113.399595</td>
      <td>0.035749</td>
      <td>1.687496</td>
    </tr>
    <tr>
      <th>128</th>
      <td>23.168789</td>
      <td>113.419960</td>
      <td>0.029593</td>
      <td>1.711373</td>
    </tr>
    <tr>
      <th>193</th>
      <td>23.162819</td>
      <td>113.418328</td>
      <td>0.028216</td>
      <td>1.801848</td>
    </tr>
    <tr>
      <th>449</th>
      <td>22.578728</td>
      <td>114.493610</td>
      <td>0.174552</td>
      <td>1.958087</td>
    </tr>
    <tr>
      <th>221</th>
      <td>22.710230</td>
      <td>113.552891</td>
      <td>0.164642</td>
      <td>1.977712</td>
    </tr>
  </tbody>
</table>
<p>831 rows × 4 columns</p>
</div>




```python
# 绘图程序
import matplotlib.pyplot as plt
for k,v in ([1,5],[5,2]):
    plt.figure('k=%d'%k)
    outliers2, inliers2 = lof(B, A, k=k, method=v)
    plt.scatter(np.array(A)[:,0],np.array(A)[:,1],s = 10,c='b',alpha = 0.5)
    plt.scatter(np.array(B)[:,0],np.array(B)[:,1],s = 10,c='green',alpha = 0.3)
    plt.scatter(outliers2[0],outliers2[1],s = 10+outliers2['local outlier factor']*100,c='r',alpha = 0.2)
    plt.title('k = %d, method = %g' % (k,v))
```


    
![png](Task02_files/Task02_34_0.png)
    



    
![png](Task02_files/Task02_34_1.png)
    

