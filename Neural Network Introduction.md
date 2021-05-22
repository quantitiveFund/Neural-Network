# Neural Network

## 1.感知机

​		感知机（Perceptron）收入多个信号，输出一个信号。至于如何判断该信号是否流过，就需要设置信号的权重和阈值。

<img src="/Users/zhoukuan/Library/Application Support/typora-user-images/image-20210519113259872.png" alt="image-20210519113259872" style="zoom:50%;" />

<center style="">图1-1 两个输入的感知机 </center>

​		在图1中，$x1、x2$是两个输入信号，$w_1 、 w_2$是其对应的权重，这里假设***阈值***为$\theta$，则有
$$
y=\begin{cases}
0 \,\,,w_1x_1+w_2x_2 \leq \theta \\
1\,\,,w_1x_1+w_2x_2 > \theta
\end{cases}
\tag{1.1}
$$
​		但是我们通常用的是***偏置$b$***，即阈值$\theta$的相反数，$\theta=-b$
$$
y=\begin{cases}
0 \,\,,b+w_1x_1+w_2x_2 \leq 0 \\
1\,\,,b+w_1x_1+w_2x_2 > 0
\end{cases}
\tag{1.2}
$$

> 感知机分类：
>
> 1、“朴素感知机”指单层网络，激活函数使用阶跃函数；
>
> 2、“多层感知机”指神经网络，激活函数使用sigmoid等。



## 2.神经网络

​		感知机能够通过权重和偏置的设定来表示一系列复杂的计算机处理，但不幸的是，权重和偏置的设定还是需要*人工*进行。而神经网络的出现就是为了解决该问题，因为神经网络的一个重要性质就是它可以**$\underline{自动地}$**从数据中学习到合适的权重参数。

<img src="/Users/zhoukuan/Library/Application Support/typora-user-images/image-20210519131933426.png" alt="image-20210519131933426" style="zoom:50%;" />

<center/>图2-1 神经网络的例子 </center>

​		如图2-1所示，该神经网络一共有三层，从左到右依次为输入层、中间层（隐藏层）、输出层。为了便于python编程，我们通常称之为第0层、第1层、第二层。

> 图2-1中实际上有权重的层只有第0层和第1层之间以及第1层和第2层之间。
>
> 因此，有些书根据权重层的数量，将图2-1称为两层神经网络。

​		第一章中的式（1.2）我们已经介绍了判断是否输出信号的条件，这里我们将其写成更简单的形式，即式（2.1）（2.2）
$$
y=h(b+w_1x_1+w_2x_2) \tag{2.1}
$$

$$
h(x)=\begin{cases}
1,x>0 \\
0,x\leq0
\end{cases} \tag{2.2}
$$

<img src="/Users/zhoukuan/Library/Application Support/typora-user-images/image-20210519133125256.png" alt="image-20210519133125256" style="zoom:80%;" />

<center>图2-2 加入偏置的感知机</center>

### 2.1激活函数

​		式（2.2）中出现的$h(x)$函数能够将输入信号的总和转换成输出信号，这样的函数我们称为**激活函数**（activation function）。激活函数的作用如其名，在于$\underline{决定是否以及如何激活输入信号的总和}。$

<img src="/Users/zhoukuan/Library/Application Support/typora-user-images/image-20210519133648301.png" alt="image-20210519133648301" style="zoom:80%;" />

<center> 图2-3 完整的感知机 </center>

​		图2-3中，$a=b+w_1x_1+w_2x_2$，表示输入信号的总和，$y=h(a)$表示经过激活函数处理转换的输出信号。

> 这里"神经元"和“节点”两个术语的含义相同，我们称$a$和$y$为节点和神经元都可以。

<center>激活函数的分类：</center>

$$
\begin{cases}
step\, function\\
\\
sigmoid\, function \\
\\
ReLU
\end{cases}
$$

#### 2.1.1 阶跃函数（step function）

<img src="/Users/zhoukuan/Library/Application Support/typora-user-images/image-20210519134834754.png" alt="image-20210519134834754" style="zoom:50%;" />

<center> 图2-4 阶跃函数 </center>

​		很直观地能看到，当输入大于0时，阶跃函数的输出值为1；当输入小于等于0时，阶跃函数的输出值为0。这也是上面介绍的感知机中用的激活函数。

阶跃函数的代码：

```python
def step_function(x):
  	y=x>0
  	return y.astype(np.int)
```



#### 2.1.2 Sigmoid函数

​		Sigmoid函数（Sigmoid function）是神经网络中常用的激活函数，其表达式为
$$
h(x)=\frac{1}{1+e^{-x}}\,\,,\,\,e=2.7182....\tag{2.3}
$$
它的图像如下图2-5所示，

<img src="/Users/zhoukuan/Library/Application Support/typora-user-images/image-20210519135452831.png" alt="image-20210519135452831" style="zoom:50%;" />

<center> 图2-5 sigmoid 函数 </center>

sigmoid函数代码实现：

```python
def sigmoid(x):
		return 1/(1+np.exp(-x))
```

​	

​	我们将上述的两个激活函数进行对比，

<img src="/Users/zhoukuan/Library/Application Support/typora-user-images/image-20210519135552946.png" alt="image-20210519135552946" style="zoom:50%;" />

<center> 图2-6 阶跃函数(虚线)与sigmoid函数 </center>

​		我们能够很直观地看到，sigmoid函数相对于阶跃函数更加平滑，变化过程更加明显，因为sigmoid函数能够返回0、0.1、0.23455、0.556765.....连续的实数，而阶跃函数只能返回0、1。

#### 2.1.3 ReLU函数

​		在神经网络的发展史上，sigmoid函数很早就开始使用了，而最近则主要是用**ReLU**（Rectified Linear Unit）函数。

​		ReLU函数在输入值小于等于0时输出0，在输入值大于0时输出其本身，即：
$$
h(x)=\begin{cases}
x , x>0 \\
0, x\leq 0
\end{cases} \tag{2.4}
$$
其图像见图2-7，

<img src="/Users/zhoukuan/Library/Application Support/typora-user-images/image-20210519140335531.png" alt="image-20210519140335531" style="zoom:50%;" />

<center> 图2-7 ReLU函数</center>

ReLU函数的代码实现：

```python
def ReLU(x):
		return np.maximum(0,x)
```



### 2.2多维数组的运算

#### 2.2.1神经网络的内积

​		我们以图2-8为例来展现神经网络的内积，

<img src="/Users/zhoukuan/Library/Application Support/typora-user-images/image-20210519143114640.png" alt="image-20210519143114640" style="zoom:50%;" />

<center> 图2-8 通过矩阵乘积来进行神经网络的计算(不含偏置)
</center>

​		在图2-8中我们令输入层$X=(x_1 \,x_2)$，权重$W=\begin{pmatrix} 1\,3\,5 \\2 \,4\, 6\end{pmatrix}$，输出层$Y=(y_1\,y_2\,y_3)$。可以得到关系式：
$$
XW=Y \tag{2.5}
$$

#### 2.2.2 三层神经网络的实现

<img src="/Users/zhoukuan/Library/Application Support/typora-user-images/image-20210519143916684.png" alt="image-20210519143916684" style="zoom:50%;" />

<center> 图2-9 三层神经网络第0层到第1层的传递过程 </center>

​		如图2-9所示，该过程表示的是完整的传递过程，其中$w_{11}^{(1)}$表示第一个输入变量$x_1$转移给第（1）层$a_1^{(1)}$的权重，$w_{12}^{(1)}$表示第二个输入变量$x_2$转移给第（1）层$a_1^{(1)}$的权重,$b_1^{(1)}$表示第（1）层$a_1^{(1)}$变量的偏置。

​		能够得到表达式(2.6):
$$
a_1^{(1)}=b_1^{(1)}+w_{11}^{(1)}x_1+w_{12}^{(1)}x_2 \tag{2.6}
$$
​		用矩阵表达即可得到式(2.7):
$$
A^{(1)}=XW^{(1)}+B^{(1)} \tag{2.7}
$$
其中，
$$
A^{(1)}=(a_1^{(1)}\,\,a_2^{(1)}\,\,a_3^{(1)}),X=(x_1\,\,x_2),\\
W^{(1)}=\begin{pmatrix}
w_{11}^{(1)}\,\,w_{21}^{(1)}\,\,w_{31}^{(1)}\\
w_{12}^{(1)}\,\,w_{22}^{(1)}\,\,w_{32}^{(1)}
\end{pmatrix}
,B^{(1)}=(b_1^{(1)}\,\,b_2^{(1)}\,\,b_3^{(1)})
$$
代码实现：

```python
#参数任意设定
#第一层的传递
x=np.array([5,4])
w1=np.array([[0.2,0.3,0.4],[0.6,0.5,0.4]])
b1=np.array([3,4,5])
a1=np.dot(x,w)+b1
z1=sigmoid(a1) #将输入信号和进行激活
```

​		得到了输入信号和还不够，要想将其输出还需要激活函数，这里我们用sigmoid函数，如图2-10所示。

<img src="/Users/zhoukuan/Library/Application Support/typora-user-images/image-20210519150710753.png" alt="image-20210519150710753" style="zoom:50%;" />

<center>图2-10 三层神经网络从输入层到第1层的完整传递过程 </center>

​		在这之后，我们开始从第1层到第2层的传递，如图2-11所示。

<img src="/Users/zhoukuan/Library/Application Support/typora-user-images/image-20210519150928526.png" alt="image-20210519150928526" style="zoom:50%;" />

<center> 图2-11 三层神经网络从第1层到第2层的完整传递过程 </center>

实现代码：

```python
w2=np.array([[0.6,0.5],[0.3,0.8],[0.2,0.7]])
b2=np.array([2,5,4])
a2=np.dot(z1,w2)+b2 #得到第1层的输入信号和
z2=sigmoid(a2)
```

​		最后是第2层到输出层的传递过程，如图2-12所示。

<img src="/Users/zhoukuan/Library/Application Support/typora-user-images/image-20210519151326925.png" alt="image-20210519151326925" style="zoom:50%;" />

<center>图2-12 三层神经网络从第2层到输出层的完整传递过程 </center>

​		需要注意的是，输出层的激活函数不再用$h()$表示，而用$\sigma()$来表示，这一层的激活函数常用的有***恒等函数、softmax函数***等，这里我们以恒等函数为例。

​		代码实现：

```python
def identity_function(x): #恒等函数
		return x
 
w3=np.array([[0.6,0.9],[0.3,0.7]])
b3=np.array([5,3])
a3=np.dot(z2,w3)+b3
y=identity_function(a3)
```

​		这里当然可以不用定义恒等函数，直接`y=a3`，之所以定义恒等函数是因为这是一个完整的神经网络中必要的步骤。



#### 2.2.3代码小结

​		至此，我们已经介绍完了神经网络的实现过程已经每一步的代码，现在我们将其整合起来。这里，我们按照神经网络的实现惯例，只把权重记为大写字母W，其他的（偏置或中间结果等）都用小写字母表示。

```python
def init_network(): #初始化神经网络各层的参数
		network={} #用于存放每一层的权重和偏置数据
    network['W1']=np.array([[0.2,0.3,0.4],[0.6,0.5,0.4]]) #2x3
    network['b1']=np.array([3,4,5])
    network['W2']=np.array([[0.6,0.5],[0.3,0.8],[0.2,0.7]]) #3x2
    network['b2']=np.array([2,5])
    network['W3']=np.array([[0.6,0.9],[0.3,0.7]]) #2x2
    network['b3']=np.array([5,3])
    
    return network
  
def forward(network, x): #向前传递
		W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=identity_function(a3)
    
    return y

network=init_network()
x=np.array([8.6,5.3])
y=forward(network,x)
print(y)
```

​		上述代码中的init_network函数储存了每一层的参数，包括权重、偏置，forward函数表示的是从输入到输出方向的传递处理。在后面进行神经网络的训练时，我们还会介绍后向（backward），从输出到输入的处理。



### 2.3输出层的设计

​		神经网络可以用在***分类问题***和***回归问题***上，不过需要根据情况改变输出层的激活函数。一般而言，回归问题用恒等函数，分类问题用softmax函数。

​		**分类问题**是数据属于哪一个类别的问题。比如，区分图像中的人是男性还是女性；而**回归问题**是根据某个输入预测一个（连续的）数值问题。比如，根据一个人的图像预测这个人的体重的问题（类似于“68.9kg这样的预测”）。



#### 2.3.1 恒等函数

​		恒等函数会将输入的数值原封不动地输出，可以用如下图像表示。

<img src="/Users/zhoukuan/Library/Application Support/typora-user-images/image-20210519161006534.png" alt="image-20210519161006534" style="zoom:50%;" />

<center>图2-13 恒等函数</center>

代码实现：

```python
def identity_function(x):
  	return x
```



#### 2.3.2softmax函数

​		分类问题中的softmax函数的表达式如下：
$$
y_k=\frac{e^{a_k}}{\sum_{i=1}^{n}e^{a_i}} \tag{2.8}
$$
​		式（2.8）表示假设输出层共有n各神经元，计算第$k$个神经元的输出$y_k$。softmax函数的分子是输入信号$a_k$的指数函数，分母是所有输入信号的指数函数的和，除此之外，softmax函数输出的值求和结果一定等于1，这样就可以表示分类问题中每一类的***概率大小***。

​		该过程可以用图2-14表示：

<img src="/Users/zhoukuan/Library/Application Support/typora-user-images/image-20210519161653525.png" alt="image-20210519161653525" style="zoom:50%;" />

<center> 图2-14 softmax函数 </center>

代码实现：

```python
def softmax(a):
  	return np.exp(a)/np.sum(np.exp(a))
```

#### sotfmax函数的注意事项：

​		softmax函数进行的是指数运算，结果可能会非常大，例如`np.exp(1000)`的输出结果为`inf`，表示无穷大。这样就会导致后续的计算无法进行。

> 计算机处理“数”时，数值必须在4字节或8字节的有限数据宽度内。
> 这意味着数存在有效位数，也就是说，可以表示的数值范围是有
> 限的。因此，会出现超大值无法表示的问题。这个问题称为***溢出***，
> 在进行计算机的运算时必须（常常）注意。

​		softmax函数的改进可以像式（2.9）这样改进：
$$
y_k=\frac{e^{a_k}}{\sum_{i=1}^{n}e^{a_i}}=\frac{Ce^{a_k}}{C\sum_{i=1}^{n}e^{a_i}}\\
=\frac{e^{a_k+lnC}}{\sum_{i=1}^{n}e^{a_i+lnC}}
=\frac{e^{a_k+C'}}{\sum_{i=1}^{n}e^{a_i+C'}}
\tag{2.9}
$$
​		即$a_k$加上任意常数$y_k$都不变，因此，为了避免溢出，$C'$我们通常取$a_k$中的最大值的相反数，即$C'=-max(a_k)$。

​		所以，优化后的softmax函数的代码实现：

```python
def softmax(x):
  	m=np.max(x)
    x-=m
    return np.exp(x)/np.sum(np.exp(x))
```

​		实例：

<img src="/Users/zhoukuan/Library/Application Support/typora-user-images/image-20210519180409351.png" alt="image-20210519180409351" style="zoom:50%;" />

​		可以看到，上面这个例子可以解释成y[0]的概率为0.018(1.8%)，y[1]的概率为0.245(24.5%)，y[2]的概率为0.737(73.7%)。从结果来看，第二个元素的概率最大，所以可以将其分为第二类。

​		但需要注意的是，即便使用了softmax函数，这些数据本身的大小关系并没有变，仅仅输出最大值也可以进行判断该分在哪一类，因此，神经网络进行分类时，输出层的softmax函数可以省略。之所以会存在softmax函数，是因为它和训练过程的损失函数有关(Loss function)。



## 3 神经网络的学习

**本节概要：**

​		本节的主要内容是神经网络的学习过程。这里的“学习”即“训练”，指的是从样本数据中习得其特征，从而自动获得最优权重配置和偏置的过程。为了使神经网络能学习，我们将导入损失函数（Loss function）的概念，而学习的目标就是让损失函数最小化，即找出使其最小的权重值和偏置。在这一部分，我们将着重介绍***梯度法***。

### 3.1 损失函数

​		损失函数（Loss function）是表示神经网络性能的“恶劣程度”的指标，即当前的神经网络对监督数据在多大程度上不拟合，在多大程度上不一致。所以，我们的目标是使得损失函数最小化。这个损失函数可以使用任意函数，但一般用**均方误差**和**交叉熵误差**等。

#### 3.1.1 均方误差

​		可以用作损失函数的函数有很多，其中最有名的是均方误差（mean squared error）。表达式如下所示：
$$
E=\frac{1}{2}\sum_k(y_k-t_k)^2
$$
​		其中$y_k$表示神经网络的输出值，$t_k$表示监督数据，$k$表示数据的维数。在识别手写数字的例子中，$y_k$和$t_k$是由如下10个元素构成的数据：

$y_k$=[0.1,  0.05,  0.6,  0.0,  0.05,  0.1,  0.0,  0.1,  0.0,  0.0]

$t_k$=[0,  0,  1,  0,  0,  0,  0,  0,  0,  0]

​		这里$k=10$,表示手写数字从0～9，$y_k$表示神经网络输出的识别每个数字的概率，$t_k$表示监督数据，即为2。

> 这里将正确解标签表示为1，其余标签表示为0的表示方法称为**one-hot表示**

均方误差实现代码：

```python
def mean_squared_error(y, t):
		return np.sum((y-t)**2)/2
```



#### 3.1.2 交叉熵误差

​		交叉熵误差(cross entropy error)也常被用于损失函数，表达式如下：
$$
E=-\sum_k t_k ln\,y_k
$$
​		由于$t_k$是one hot表示法，只有正确分类的标签为1，其余为0，所以上式中只需算标签为1的部分。