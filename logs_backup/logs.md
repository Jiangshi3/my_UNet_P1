trained_model_12-14： num_epochs=40； resize(512, 512); ,batch_size=2; lr=0.00001;  weight = [1, 50]
> 结果： 在epoch35左右，验证集有波动；有可能是过拟合了？
> 根据test生成数据，发现效果很差，可能是loss改的不好；还是得让L1损失占据主要部分？



trained_model_12-14-2: num_epochs=30； resize(512, 512); batch_size=4;  lr=0.00001;  weight = [1, 20]   

> 到15个epoch爆内存了； 根据损失图发现lr有点小了



trained_model_12-15-毙1: num_epochs=30； resize(512, 512); batch_size=3;  lr=0.0001;  weight = [1, 20]   

> lr太大了；中断，没让它跑完；



trained_model_12-15: num_epochs=30； resize(512, 512); batch_size=3;  lr=0.00002;  weight = [1, 20] 

> 这里开始设置了best_loss, 根据best_loss来保存模型参数；
> 观察loss，发现lr还是有点大？



trained_model_12-16: num_epochs=30； resize(512, 512); batch_size=3;  lr=0.00001;  weight = [1, 20] 

> 观察loss，发现损失曲线还ok；但是测试图片效果不好，是不是resize512不如256呢？？？  

<img src="D:/笔记/md/imgs/image-20231225094612482.png" alt="image-20231225094612482" style="zoom:33%;" />

trained_model_12-16-2: num_epochs=30； resize(256, 256); batch_size=4;  lr=0.00001;  weight = [1, 20]

> 此时的L1lose比MS_SSIM大了五倍多(一开始的loss)  【batch_size需要增大一些吗？？】
>
> 结果：loss曲线还可以；测试效果不行

<img src="D:/笔记/md/imgs/image-20231216165341814.png" alt="image-20231216165341814" style="zoom:50%;" />



> 总结发现调整损失函数后的训练效果都很差(看来还是让L1Loss发挥最主要的作用，而不是让MS_SSIM与L1占的权重相近)；
> 现在还暂时不确定resize(512, 512)和resize(256, 256)哪个更好，感觉还是应该resize(512, 512)更好一些吧？？



trained_model_12-16-3: num_epochs=30； resize(512, 512); batch_size=3;  lr=0.00001;  weight = [0.8, 0.2]

<img src="D:/笔记/md/imgs/image-20231216191655204.png" alt="image-20231216191655204" style="zoom: 50%;" />

> 效果还是不太行，要不测试一下只采用L1损失？？
>
> 1、难道还是resize(512, 512)不行？？  
>
>  2、先测试一下只采用L1损失，再测试使用resize(256，256)



trained_model_12-16-4:  num_epochs=20； resize(512, 512); batch_size=3;  lr=0.00001;  weight = [1.0,  0]

> 只使用L1损失

<img src="D:/笔记/md/imgs/image-20231216210248439.png" alt="image-20231216210248439" style="zoom: 80%;" />

> 实验结果太差了？？？ 【sb，loss这么高，效果肯定差呀】

> next:
>
> -   使用resize(256, 256)???
>
> - （采用最好结果的超参数跑一遍先？） 找到之前最好的超参数！！  现在的结果太差
>
>   ​    （之前只使用的L2损失；  修改了loss函数后的效果都不好；还是要用回之前的loss。唉）

### 12月17号

trained_model_12-17-1:  num_epochs=20； resize(512, 512); batch_size=3;  lr=0.0001;  L2_loss

> 训练了两个epoch，发现loss下降太大（一下子下降了一半多）；

trained_model_12-17-2:  num_epochs=20； resize(512, 512); batch_size=3;  lr=0.00001;  L2_loss

<img src="D:/笔记/md/imgs/image-20231217201604717.png" alt="image-20231217201604717" style="zoom:50%;" />

> 损失曲线还可以，但是没有测试结果并没有达到以前的（这次epoch比较少）

#### trained_model_12-17-2-resume

trained_model_12-17-2-resume： 加载训练参数，又多训练了10个epoch，效果微乎其微。。

> 要想得到继续训练的连续曲线，不要删除之前的log，在同一个文件夹中，然后设置继续训练开始的epoch等于之前结束的epoch，结束的epoch等于之前结束的加上多训练的轮数，就能得到一个连续的loss曲线了。（设置epoch和num_epochs）

<img src="D:/笔记/md/imgs/image-20231217212457771.png" alt="image-20231217212457771" style="zoom:25%;" />

> 难道还是resize(256,256)效果最好吗？

trained_model_12-17-3：num_epochs=20； resize(256, 256); batch_size=4;  lr=0.00001;   L2_loss

> 依托答辩。。。。 why??? 呜呜呜

<img src="D:/笔记/md/imgs/image-20231217223619997.png" alt="image-20231217223619997" style="zoom: 50%;" />

### 12月18号

trained_model_12-18：num_epochs=20； resize(256, 256); batch_size=4;  lr=0.0001;   L2_loss

> （这个感觉就是之前最好效果的超参数了。。。） ----效果没有那么差了，但也不好

<img src="D:/笔记/md/imgs/image-20231218101134756.png" alt="image-20231218101134756" style="zoom:50%;" />

再给它多训练5轮；

<img src="D:/笔记/md/imgs/image-20231218102806245.png" alt="image-20231218102806245" style="zoom: 50%;" />

再多训练10轮：

<img src="D:/笔记/md/imgs/image-20231218121316638.png" alt="image-20231218121316638" style="zoom: 50%;" />

> 效果还可以，但是还没达到之前最好的效果。
>
> （多训练几轮的效果就会好一些，难道之前轮数太少了？？虽然loss下降的很少）

#### trained_model_12-18-2

trained_model_12-18-2：num_epochs=40； resize(256, 256); batch_size=4;  lr=0.00005;   L2_loss

<img src="D:/笔记/md/imgs/image-20231218151920481.png" alt="image-20231218151920481" style="zoom:50%;" />

> 效果好一些了

trained_model_12-18-3：num_epochs=【30~40】； resize(512, 512); batch_size=3;  lr=0.00001;  L2_loss

​	继续训练trained_model_12-17-2-resume；这里已经训练了30个epoch；

<img src="D:/笔记/md/imgs/image-20231218163336512.png" alt="image-20231218163336512" style="zoom:50%;" />

> 效果一般 （发现lr=0.0001是效果最好的，难道我就不需要去管loss曲线？让他前期可以下降很快？）

trained_model_12-18-4

回到trained_model_12-17-1:  num_epochs=20； resize(512, 512); batch_size=3;  lr=0.0001;  L2_loss，这里我认为是比较好的超参数，但是看到loss下降太快就没有去训练了；这次把他训练看一看。

<img src="D:/笔记/md/imgs/image-20231218200300562.png" alt="image-20231218200300562" style="zoom:50%;" />

> 效果一般吧。。



改loss吗？？？？

改网络架构吗？？？

### 12月23号

trained_model_12-23：num_epochs=20； resize(512, 512); batch_size=2;  lr=0.0001;  L2_loss

> 这个是trained_model_12-05.pth的超参数(这个效果还可以)；它的epoch比较少只有20，再重新训练一遍；
>
> 【但我感觉它的batchsize只有2，lr=0.0001, 相对来猜测：batchsize有点小，lr有点大。】

<img src="D:/笔记/md/imgs/image-20231223112816630.png" alt="image-20231223112816630" style="zoom: 80%;" />

> 最后的损失值为4左右；这个效果对于raw还挺好，对于challenge没有太好；

trained_model_12-23-resume：num_epochs=【20~30】； resize(512, 512); batch_size=2;  lr=0.00005;  L2_loss

> 看验证集的loss是波动状态，说明lr有点大，把lr调小一些继续训练；

<img src="D:/笔记/md/imgs/image-20231223213825318.png" alt="image-20231223213825318" style="zoom:50%;" />

> 还可以吧，challenge不太好；

trained_model_12-23-2：num_epochs=20； resize(512, 512); batch_size=2;  lr=0.00001;  L2_loss

<img src="D:/笔记/md/imgs/image-20231223152042869.png" alt="image-20231223152042869" style="zoom:50%;" />

> 最低点的损失值是7.多，lr小了，训练的轮数也应该多一些。再来10轮；

trained_model_12-23-2-resume：num_epochs=【20~30】； resize(512, 512); batch_size=2;  lr=0.00001;  L2_loss

<img src="D:/笔记/md/imgs/image-20231223161629721.png" alt="image-20231223161629721" style="zoom:50%;" />

> 最低点的loss_value=7.左右；lr还是太小了，训练不动了； （lr增大，再训练几轮看看？）
>
> 【难道一开始采用较大的学习率，后续训练不动后，resume时更换较小的学习率？？】

trained_model_12-23-2-resume-2：num_epochs=【30~35】； resize(512, 512); batch_size=2;  lr=0.00005;  L2_loss

> 最低点的loss_value=5.左右；  效果挺差的；

<img src="D:/笔记/md/imgs/image-20231223170001121.png" alt="image-20231223170001121" style="zoom:50%;" />

trained_model_12-23-2-resume-2：num_epochs=【35~40】； resize(512, 512); batch_size=2;  lr=0.00005;  L2_loss

<img src="D:/笔记/md/imgs/image-20231223174130891.png" alt="image-20231223174130891" style="zoom:50%;" />

> 最低点的loss_value=4.5左右；  效果没有太好，也没有很差；



### 12月24号

trained_model_12-24-1：num_epochs=20； resize(512, 512); batch_size=2;  lr=0.00005;  L2_loss

<img src="D:/笔记/md/imgs/image-20231224120520866.png" alt="image-20231224120520866" style="zoom:50%;" />

> 这个要继续resume训练几轮吗？继续训练的超参数该如何设置呢？

trained_model_12-24-1：num_epochs=【20，30】； resize(512, 512); batch_size=2;  lr=0.00002;  L2_loss



【这里改了一下结构(把跳跃连接的拼接改为相加)重新去训练，对比两个结构的差异：】

弄来：trained_model_12-23：num_epochs=20； resize(512, 512); batch_size=2;  lr=0.0001;  L2_loss；按照另一个模型的超参数去训练；

> 另一个模型：trained_model_12-24-1：num_epochs=30； resize(512, 512); batch_size=2;  lr=0.0001;  L2_loss；
>
> ​					trained_model_12-24-1：num_epochs=【30，40】； resize(512, 512); batch_size=2;  lr=0.00005;  L2_loss

**resume**：trained_model_12-23-resume-1：num_epochs=【20,30】； resize(512, 512); batch_size=2;  lr=0.0001;  L2_loss



​			：trained_model_12-23-resume-1：num_epochs=【30,40】； resize(512, 512); batch_size=2;  lr=0.00005;  L2_loss

<img src="D:/笔记/md/imgs/image-20231224225549580.png" alt="image-20231224225549580" style="zoom:33%;" />

> 和修改过的模型训练结果差不多。。。



