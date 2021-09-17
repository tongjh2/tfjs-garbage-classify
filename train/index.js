const tf = require("@tensorflow/tfjs-node")
const getData = require("./data")

const TRAIN_DIR = '垃圾分类/train'
const OUTPUT_DIR = 'output'
// const MOBILE_NET_URL = "https://ai-sample.oss-cn-hangzhou.aliyuncs.com/pipcook/models/mobilenet/web_model/model.json";
const MOBILE_NET_URL = 'http://ai-sample.oss-cn-hangzhou.aliyuncs.com/pipcook/models/mobilenet/web_model/model.json';
//定义模型
const defineModel = async(classes)=>{
    const mobileNet = await tf.loadLayersModel(MOBILE_NET_URL)

    mobileNet.summary();

    console.log(mobileNet.layers.map((l, i) => [l.name, i]))

    const model = tf.sequential();
    
    for (let i = 0; i <= 10; i += 1) {
        let layer = mobileNet.layers[i];
        layer.trainable = false; //设置这些层不参与训练
        model.add(layer)
    }

    model.add(tf.layers.flatten())

    model.add(tf.layers.dense({
        units: 10, //神经元个数，超参数
        activation: 'relu' //激活函数，解决非线性问题
    }));

    model.add(tf.layers.dense({
        units: classes.length, //类的个数
        activation: 'softmax' //多分类
    }));

    model.summary();
    return model;
}

//训练模型
const trainModel = async(model,ds)=>{
    // //训练模型
    model.compile({
        loss: 'sparseCategoricalCrossentropy',
        optimizer: tf.train.adam(),
        metrics: ['acc']
    })
    await model.fitDataset(ds, {
        epochs: 20
    });
    await model.save(`file://${process.cwd()}/${OUTPUT_DIR}`);
}

const main = async () => {
    //加载数据
    const {
        ds,
        classes
    } = await getData(TRAIN_DIR, OUTPUT_DIR);
    // console.log(ds, classes)

    //定义模型
    const model = await defineModel(classes)

    //训练模型
    await trainModel(model,ds)    
}

main();