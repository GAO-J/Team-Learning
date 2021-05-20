# 异常检测——高维数据异常检测

**参考**：

https://github.com/datawhalechina/team-learning-data-mining/tree/master/AnomalyDetection

https://zhuanlan.zhihu.com/p/36822575



**思维导图：**

<img src="D:\DataWhale\OutlierAnalysis\Task02\img\QQ图片20210520223019.jpg" alt="png" style="zoom:50%;" />



**主要内容包括：**

* **Feature Bagging**

- **孤立森林** 

  

[TOC]

## 1、引言

在实际场景中，很多数据集都是多维度的。随着维度的增加，数据空间的大小（体积）会以指数级别增长，使数据变得稀疏，这便是维度诅咒的难题。维度诅咒不止给异常检测带来了挑战，对距离的计算，聚类都带来了难题。例如基于邻近度的方法是在所有维度使用距离函数来定义局部性，但是，在高维空间中，所有点对的距离几乎都是相等的（距离集中），这使得一些基于距离的方法失效。在高维场景下，一个常用的方法是子空间方法。

集成是子空间思想中常用的方法之一，可以有效提高数据挖掘算法精度。集成方法将多个算法或多个基检测器的输出结合起来。其基本思想是一些算法在某些子集上表现很好，一些算法在其他子集上表现很好，然后集成起来使得输出更加鲁棒。集成方法与基于子空间方法有着天然的相似性，子空间与不同的点集相关，而集成方法使用基检测器来探索不同维度的子集，将这些基学习器集合起来。



下面来介绍两种常见的集成方法：

## 2、Feature Bagging

Feature Bagging，基本思想与bagging相似，只是对象是feature。feature bagging属于集成方法的一种。集成方法的设计有以下两个主要步骤：

**1.选择基检测器**。这些基本检测器可以彼此完全不同，或不同的参数设置，或使用不同采样的子数据集。Feature bagging常用lof算法为基算法。下图是feature bagging的通用算法：

![image-20210104144520790](D:\DataWhale\team-learning-data-mining-master\AnomalyDetection\img\image-20210104144520790.png)

**2.分数标准化和组合方法**：不同检测器可能会在不同的尺度上产生分数。例如，平均k近邻检测器会输出原始距离分数，而LOF算法会输出归一化值。另外，尽管一般情况是输出较大的异常值分数，但有些检测器会输出较小的异常值分数。因此，需要将来自各种检测器的分数转换成可以有意义的组合的归一化值。分数标准化之后，还要选择一个组合函数将不同基本检测器的得分进行组合，最常见的选择包括平均和最大化组合函数。

下图是两个feature bagging两个不同的组合分数方法：

![image-20210105140222697-1609839336763](D:\DataWhale\team-learning-data-mining-master\AnomalyDetection\img\image-20210105140222697-1609839336763.png)

​																			(广度优先)

![image-20210105140242611](D:\DataWhale\team-learning-data-mining-master\AnomalyDetection\img\image-20210105140242611.png)

​																			（累积求和）

​													

基探测器的设计及其组合方法都取决于特定集成方法的特定目标。很多时候，我们无法得知数据的原始分布，只能通过部分数据去学习。除此以外，算法本身也可能存在一定问题使得其无法学习到数据完整的信息。这些问题造成的误差通常分为偏差和方差两种。

**方差**：是指算法输出结果与算法输出期望之间的误差，描述模型的离散程度，数据波动性。

**偏差**：是指预测值与真实值之间的差距。即使在离群点检测问题中没有可用的基本真值



## 3、Isolation Forests

孤立森林（Isolation Forest）算法是周志华教授等人于2008年提出的异常检测算法，是机器学习中少见的专门针对异常检测设计的算法之一，方法因为该算法时间效率高，能有效处理高维数据和海量数据，无须标注样本，在工业界应用广泛。

孤立森林属于非参数和无监督的算法，既不需要定义数学模型也不需要训练数据有标签。孤立森林查找孤立点的策略非常高效。假设我们用一个随机超平面来切割数据空间，切一次可以生成两个子空间。然后我们继续用随机超平面来切割每个子空间并循环，直到每个子空间只有一个数据点为止。直观上来讲，那些具有高密度的簇需要被切很多次才会将其分离，而那些低密度的点很快就被单独分配到一个子空间了。孤立森林认为这些很快被孤立的点就是异常点。

用四个样本做简单直观的理解，d是最早被孤立出来的，所以d最有可能是异常。

![img](D:\DataWhale\team-learning-data-mining-master\AnomalyDetection\img\v2-bb94bcf07ced88315d0a5de47677200e_720w.png)



怎么来切这个数据空间是孤立森林的核心思想。因为切割是随机的，为了结果的可靠性，要用集成（ensemble）的方法来得到一个收敛值，即反复从头开始切，平均每次切的结果。孤立森林由t棵孤立的数组成，每棵树都是一个随机二叉树，也就是说对于树中的每个节点，要么有两个孩子节点，要么一个孩子节点都没有。树的构造方法和随机森林(random forests)中树的构造方法有些类似。流程如下：

1)      从训练数据中随机选择一个样本子集，放入树的根节点；

2)      随机指定一个属性，随机产生一个切割点V，即属性A的最大值和最小值之间的某个数；

3)      根据属性A对每个样本分类，把A小于V的样本放在当前节点的左孩子中，大于等于V的样本放在右孩子中，这样就形成了2个子空间；

4)      在孩子节点中递归步骤2和3，不断地构造左孩子和右孩子，直到孩子节点中只有一个数据，或树的高度达到了限定高度。

获得t棵树之后，孤立森林的训练就结束，就可以用生成的孤立森林来评估测试数据。



孤立森林检测异常的假设是：异常点一般都是非常稀有的，在树中会很快被划分到叶子节点，因此可以用叶子节点到根节点的路径长度来判断一条记录是否是异常的。和随机森林类似，孤立森林也是采用构造好的所有树的平均结果形成最终结果的。在训练时，每棵树的训练样本是随机抽样的。从孤立森林的树的构造过程看，它不需要知道样本的标签，而是通过阈值来判断样本是否异常。因为异常点的路径比较短，正常点的路径比较长，孤立森林根据路径长度来估计每个样本点的异常程度。	

路径长度计算方法：

![image-20210103183909407](D:\DataWhale\team-learning-data-mining-master\AnomalyDetection\img\image-20210103183909407.png)



孤立森林也是一种基于子空间的方法，不同的分支对应于数据的不同局部子空间区域，较小的路径对应于孤立子空间的低维

## 4、总结

1.feature bagging可以降低方差	

2.孤立森林的优势在于：

- 计算成本相比基于距离或基于密度的算法更小。
- 具有线性的时间复杂度。
- 在处理大数据集上有优势。

孤立森林不适用于超高维数据，因为鼓励森林每次都是随机选取维度，如果维度过高，则会存在过多噪音。



## 5、练习

### 5.1 思考题

**1.思考题：feature bagging为什么可以降低方差？**

集成学习中，增加模型数量，可以有效提升综合模型的泛化能力，对于新的数据分布也会有比较好的适应性，总体到模型的输出比较稳定，可以有效降低方差。

<img src="https://img-blog.csdnimg.cn/20190917151718716.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2ODExMzc3,size_16,color_FFFFFF,t_70" alt="png" style="zoom:50%;" />

> 作者：过拟合
> 链接：https://www.zhihu.com/question/26760839/answer/40337791
> 来源：知乎
>
> Bagging对样本重采样，对每一重采样得到的子样本集训练一个模型，最后取平均。由于子样本集的相似性以及使用的是同种模型，因此各模型有近似相等的bias和variance（事实上，各模型的分布也近似相同，但不独立）。由于![[公式]](https://www.zhihu.com/equation?tex=E%5B%5Cfrac%7B%5Csum+X_i%7D%7Bn%7D%5D%3DE%5BX_i%5D)，所以bagging后的bias和单个子模型的接近，一般来说不能显著降低bias。另一方面，若各子模型独立，则有![[公式]](https://www.zhihu.com/equation?tex=Var%28%5Cfrac%7B%5Csum+X_i%7D%7Bn%7D%29%3D%5Cfrac%7BVar%28X_i%29%7D%7Bn%7D)，此时可以显著降低variance。若各子模型完全相同，则![[公式]](https://www.zhihu.com/equation?tex=Var%28%5Cfrac%7B%5Csum+X_i%7D%7Bn%7D%29%3DVar%28X_i%29)
>
> ，此时不会降低variance。bagging方法得到的各子模型是有一定相关性的，属于上面两个极端状况的中间态，因此可以一定程度降低variance。为了进一步降低variance，Random forest通过随机选取变量子集做拟合的方式de-correlated了各子模型（树），使得variance进一步降低。
>
> （用公式可以一目了然：设有i.d.的n个随机变量，方差记为![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%5E2)，两两变量之间的相关性为![[公式]](https://www.zhihu.com/equation?tex=%5Crho)，则![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Csum+X_i%7D%7Bn%7D)的方差为![[公式]](https://www.zhihu.com/equation?tex=%5Crho%2A%5Csigma%5E2%2B%281-%5Crho%29%2A%5Csigma%5E2%2Fn)
>
> ，bagging降低的是第二项，random forest是同时降低两项。详见ESL p588公式15.1）
>
> boosting从优化角度来看，是用forward-stagewise这种贪心法去最小化损失函数![[公式]](https://www.zhihu.com/equation?tex=L%28y%2C+%5Csum_i+a_i+f_i%28x%29%29)。例如，常见的AdaBoost即等价于用这种方法最小化exponential loss：![[公式]](https://www.zhihu.com/equation?tex=L%28y%2Cf%28x%29%29%3Dexp%28-yf%28x%29%29)。所谓forward-stagewise，就是在迭代的第n步，求解新的子模型f(x)及步长a（或者叫组合系数），来最小化![[公式]](https://www.zhihu.com/equation?tex=L%28y%2Cf_%7Bn-1%7D%28x%29%2Baf%28x%29%29)，这里![[公式]](https://www.zhihu.com/equation?tex=f_%7Bn-1%7D%28x%29)
>
> 是前n-1步得到的子模型的和。因此boosting是在sequential地最小化损失函数，其bias自然逐步下降。但由于是采取这种sequential、adaptive的策略，各子模型之间是强相关的，于是子模型之和并不能显著降低variance。所以说boosting主要还是靠降低bias来提升预测精度。



**2.思考题：feature bagging存在哪些缺陷，有什么可以优化的idea？**

要训练多个弱学习器，计算开销大。

优化：在随机采样训练若学习器前，对样本训练集进行KNN聚类，筛选明显存在异常值的样本集，再进行弱学习器的训练。



### 5.2 feature bagging


```python
from pyod.models.feature_bagging import FeatureBagging
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
```


```python
# 使用生成样本数据pyod.utils.data.generate_data()

contamination = 0.1  # percentage of outliers
n_train = 200  # number of training points
n_test = 100  # number of testing points

X_train, y_train, X_test, y_test = generate_data(n_train=n_train, n_test=n_test, contamination=contamination,behaviour="old")
```


```python
# 初始化检测器，拟合模型，然后进行预测。

# train FeatureBagging detector
clf_name = 'FeatureBagging'


clf = FeatureBagging()

clf.fit(X_train)

# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

# get the prediction on the test data
y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores
```


```python
# 使用ROC和Precision @ Rank n评估预测pyod.utils.data.evaluate_print()。

from pyod.utils.data import evaluate_print
# evaluate and print the results
print("\nFeatureBagging On Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nFeatureBagging On Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)
# 在培训和测试数据上查看示例输出。
```


    FeatureBagging On Training Data:
    FeatureBagging ROC:0.9747, precision @ rank n:0.8
    
    FeatureBagging On Test Data:
    FeatureBagging ROC:0.9611, precision @ rank n:0.8



```python
# 通过可视化所有示例中包含的功能来生成可视化。
visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
              y_test_pred, show_figure=True, save_figure=True)
```


​    
![png](img/Task05_6_0.png)
​    


### 5.3 isolation forests


```python
from pyod.models.iforest import IForest
```


```python
# train Isolation Forests detector
clf_name = 'IsolationForest'
clf = IForest()
clf.fit(X_train)

# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

# get the prediction on the test data
y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores
```


```python
# 使用ROC和Precision @ Rank n评估预测pyod.utils.data.evaluate_print()。

from pyod.utils.data import evaluate_print
# evaluate and print the results
print("\nIsolationForest On Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nIsolationForest On Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)
```


    IsolationForest On Training Data:
    IsolationForest ROC:0.9631, precision @ rank n:0.7
    
    IsolationForest On Test Data:
    IsolationForest ROC:0.9656, precision @ rank n:0.8



```python
# 通过可视化所有示例中包含的功能来生成可视化。
visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
              y_test_pred, show_figure=True, save_figure=True)
```


![png](img/Task05_11_0.png)
​    

