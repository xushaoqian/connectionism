# connectionism
connectionism ann

大脑真实的学习过程简单粗暴，就是根据hebb法则，把同时激活的感知，行为，反馈神经元加强连接，这样当你下次遇到相同场景时，就知道结果了，这是一个记忆过程，并不是一个训练过程。
这种hebb记忆方式不同于bp反向训练过程，需要海量的多层神经元作为基础，其中隐藏着大量可用的神经联结，就像已经训练好权重的神经元，我们只要找到这些神经元，和结果联结起来就可以了。
我尝试实现这种挑选算法，在特征层后面再增加挑选层，挑选层末尾有海量的神经元，每个神经元代表一种特定场景，如果这个神经元以前从来没激活过，这次激活了，就把它和结果联结起来。
