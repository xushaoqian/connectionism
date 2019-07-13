// 我在思考一种模拟神经元联结学习的算法，不同于bp反向学习
// 在我看来人脑底层是神经网络，上层是贝叶斯分类联结
// 在大量随机神经元联结的情况下，隐藏着许多有用的神经元结构，相当于已经训练好的权重
// 我们只要想办法找到这些有用的结构，选择神经网络最后一层，找出有用的神经元，加强这些联结就可以了
// 本质上避免局部最少值问题，但需要海量的随机神经元基础
// 本质上提高迁移学习能力，只需要联结不同的反馈接触点
class BPNet {
    constructor(layernum, n, fn, fd, miu, iter, eps) {
        if (!(n instanceof Array)) {
            throw '参数错误'
        }
        if (!n.length == layernum) {
            throw '参数错误'
        }
        this.layernum = layernum
        this.n = n
        //输出函数
        if (!fn) {
            this.fn = function (x) {
                return 1.0 / (1.0 + Math.exp(-x))
            }
        } else {
            this.fn = fn
        }
        //误差函数
        if (!fd) {
            this.fd = function (x) {
                return x * (1 - x)
            }
        } else {
            this.fd = fd
        }
        this.w = new Array()//权值矩阵
        this.b = new Array() //阈值矩阵
        this.miu = miu || 0.5 //学习速率
        this.iter = iter || 500 //迭代次数
        this.e = 0.0 //误差
        this.eps = eps || 0.0001
        for (let l = 1; l < this.layernum; l++) {
            let item = new Array()
            let bitem = new Array()
            for (let j = 0; j < n[l]; j++) {
                let temp = new Array()
                for (let i = 0; i < n[l - 1]; i++) {
                    temp[i] = Math.random()
                }
                item.push(temp)
                bitem.push(Math.random())
            }
            this.w[l] = item
            this.b[l] = bitem
        }
    }
    //预测函数
    forward(x) {
        let y = new Array()
        y[0] = x
        for (let l = 1; l < this.layernum; l++) {
            y[l] = new Array()
            for (let j = 0; j < this.n[l]; j++) {
                let u = 0.0
                for (let i = 0; i < this.n[l - 1]; i++) {
                    u = u + this.w[l][j][i] * y[l - 1][i]
                }
                u = u + this.b[l][j]
                y[l][j] = this.fn(u)
            }
        }
        return y
    }
    //计算误差
    calcdelta(d, y) {
        let delta = new Array()
        let last = new Array()
        for (let j = 0; j < this.n[this.layernum - 1]; j++) {
            last[j] = (d[j] - y[this.layernum - 1][j]) * this.fd(y[this.layernum - 1][j])
        }
        delta[this.layernum - 1] = last
        for (let l = this.layernum - 2; l > 0; l--) {
            delta[l] = new Array()
            for (let j = 0; j < this.n[l]; j++) {
                delta[l][j] = 0.0
                for (let i = 0; i < this.n[l + 1]; i++) {
                    delta[l][j] += delta[l + 1][i] * this.w[l + 1][i][j]
                }
                delta[l][j] = this.fd(y[l][j]) * delta[l][j]
            }
        }
        return delta
    }
    //调整权值和阈值
    update(y, delta) {
        for (let l = 0; l < this.layernum; l++) {
            for (let j = 0; j < this.n[l]; j++) {
                for (let i = 0; i < this.n[l - 1]; i++) {
                    this.w[l][j][i] += this.miu * delta[l][j] * y[l - 1][i]
                    this.b[l][j] += this.miu * delta[l][j]
                }
            }
        }
    }
    //样本训练
    train(x, d) {
        for (let p = 0; p < this.iter; p++) {
            this.e = 0
            for (let i = 0; i < x.length; i++) {
                let y = this.forward(x[i])
                let delta = this.calcdelta(d[i], y)
                this.update(y, delta)
                let ep = 0.0
                let l1 = this.layernum - 1
                for (let l = 0; l < this.n[l1]; l++) {
                    ep += (d[i][l] - y[l1][l]) * (d[i][l] - y[l1][l])
                }
                this.e += ep / 2.0
            }
            if (this.e < this.eps) {
                break;
            }
        }
    }
}

let x = [[0, 0], [0, 1], [1, 0], [1, 1], [0.1, 0.1], [0.1, 0.9], [0.94, 0.11], [0.92, 0.83], [0.23, 0.13], [0.13, 0.98], [0.92, 0.11], [0.92, 0.91]]//输入样本
let d = [[0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1], [0]]


// 模拟大脑联结学习时，初始化随机权重【2，360，60】，最后的60个随机特征是所有随机特征集。
// 训练不同的任务时，根据比较正负样本得到的特征，筛选出有用的激活特征。
// 预测不同的任务时，根据需要选择60个特征中的一部分有用特征进行预测。
let bp = new BPNet(3, [2, 6, 1], undefined, undefined, 0.5, 5000, 0.0001)
bp.train(x, d)
let y = bp.forward([0, 1])
console.log(y[2][0])
let y2 = bp.forward([0, 0])
console.log(y2[2][0])
let y3 = bp.forward([1, 1])
console.log(y3[2][0])
let y4 = bp.forward([1, 0])
console.log(y4[2][0])
