const fs = require("fs")
const tf = require("@tensorflow/tfjs-node")

//图片转为张量
const imgToX = (imgPath)=>{
    const buffer = fs.readFileSync(imgPath)

    //清楚中间变量，节省内存
    return tf.tidy(()=>{
        //张量
        const imgTs = tf.node.decodeImage(new Uint8Array(buffer));
        //图片resize
        const imgTsResized = tf.image.resizeBilinear(imgTs,[224,224])
        //归一化到[-1,1]之间
        //224*224*RGB*1张
        return imgTsResized.toFloat().sub(255/2).div(255/2).reshape([1,224,224,3])
    })
}

// 分批次读取数据
const batchReadData = async(data)=>{
    const ds = tf.data.generator(function*(){
        const count = data.length;
        const batchSize = 32;

        for(let i=0;i<count;i+=batchSize){
            const end = Math.min(i + batchSize,count);
            yield tf.tidy(()=>{
                const inputs = [];
                const labels = []
                for(let j=i;j<end;i++){
                    const {imgPath,index} = data[j]
                    const x= imgToX(imgPath)
                    inputs.push(x)
                    labels.push(index);
                }
                const xs = tf.concat(inputs)
                const ys = tf.tensor(labels)
                return {
                    xs,
                    ys
                }
            })
        }
    })

    return ds;
}

//加载数据
const getData =  async(trainDir,outputDir)=>{
    const classes = fs.readdirSync(trainDir).filter(n=>!n.includes("."))
    fs.writeFileSync(`${outputDir}/classes.json`,JSON.stringify(classes))

    const data = [];
    classes.forEach((dir,index)=>{
        fs.readdirSync(`${trainDir}/${dir}`)
        .filter(n=>n.match(/jpg$/))
        .slice(0,100)
        .forEach(filename=>{
            console.log('读取图片：',dir,filename)
            const imgPath = `${trainDir}/${dir}/${filename}`;
            data.push({ imgPath,index})
        })
    });

    //打乱数据
    tf.util.shuffle(data);

    const ds = await batchReadData(data);
    return {
        ds,
        classes
    }
}

module.exports = getData;