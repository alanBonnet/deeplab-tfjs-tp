import tf,{ zeros } from '@tensorflow/tfjs';
import deeplab,{ load } from '@tensorflow-models/deeplab';
const loadModel = async () => {
    const modelName = 'pascal';   // set to your preferred model, either `pascal`, `cityscapes` or `ade20k`
    const quantizationBytes = 2;  // either 1, 2 or 4
    return await load({ base: modelName, quantizationBytes });
};
import img from './imagenprueba.jpg';
const imgTensor = tf.browser.FromPixels(img)
console.log(imgTensor)
const input = zeros([227, 500, 3]);
// ...

loadModel()
    .then((model) => {console.log("pasado por el modelo xd");return model.segment(input,)})
    .then(
        ({ legend }) =>
            console.log(`The predicted classes are ${JSON.stringify(legend)}`));


    