"use strict";
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var tfconv = require("@tensorflow/tfjs-converter");
var tf = require("@tensorflow/tfjs-core");
var utils_1 = require("./utils");
exports.getColormap = utils_1.getColormap;
exports.getLabels = utils_1.getLabels;
exports.getURL = utils_1.getURL;
exports.toSegmentationImage = utils_1.toSegmentationImage;
var version_1 = require("./version");
exports.version = version_1.version;
/**
 * Initializes the DeepLab model and returns a `SemanticSegmentation` object.
 *
 * @param input ::
 *     `ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement`
 *
 *  The input image to feed through the network.
 *
 * @param config :: `ModelConfig`
 *
 * The configuration for the model with any of the following attributes:
 *
 *   * quantizationBytes (optional) :: `QuantizationBytes`
 *
 *      The degree to which weights are quantized (either 1, 2 or 4).
 *      Setting this attribute to 1 or 2 will load the model with int32 and
 *      float32 compressed to 1 or 2 bytes respectively.
 *      Set it to 4 to disable quantization.
 *
 *   * base (optional) :: `ModelArchitecture`
 *
 *      The type of model to load (either `pascal`, `cityscapes` or `ade20k`).
 *
 *   * modelUrl (optional) :: `string`
 *
 *      The URL from which to load the TF.js GraphModel JSON.
 *      Inferred from `base` and `quantizationBytes` if undefined.
 *
 * @return The initialized `SemanticSegmentation` object
 */
function load(modelConfig) {
    if (modelConfig === void 0) { modelConfig = {
        base: 'pascal',
        quantizationBytes: 2
    }; }
    return __awaiter(this, void 0, void 0, function () {
        var graphModel, deeplab;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (tf == null) {
                        throw new Error("Cannot find TensorFlow.js." +
                            " If you are using a <script> tag, please " +
                            "also include @tensorflow/tfjs on the page before using this model.");
                    }
                    if (modelConfig.base) {
                        if (['pascal', 'cityscapes', 'ade20k'].indexOf(modelConfig.base) === -1) {
                            throw new Error("SemanticSegmentation cannot be constructed " +
                                ("with an invalid base model " + modelConfig.base + ". ") +
                                "Try one of 'pascal', 'cityscapes' and 'ade20k'.");
                        }
                        if ([1, 2, 4].indexOf(modelConfig.quantizationBytes) === -1) {
                            throw new Error("Only quantization to 1, 2 or 4 bytes is supported.");
                        }
                    }
                    else if (!modelConfig.modelUrl) {
                        throw new Error("SemanticSegmentation can be constructed either by passing " +
                            "the weights URL or one of the supported base model names from " +
                            "'pascal', 'cityscapes' and 'ade20k'," +
                            "together with the degree of quantization (either 1, 2 or 4)." +
                            "Aborting, since neither has been provided.");
                    }
                    return [4 /*yield*/, tfconv.loadGraphModel(modelConfig.modelUrl ||
                            utils_1.getURL(modelConfig.base, modelConfig.quantizationBytes))];
                case 1:
                    graphModel = _a.sent();
                    deeplab = new SemanticSegmentation(graphModel, modelConfig.base);
                    return [2 /*return*/, deeplab];
            }
        });
    });
}
exports.load = load;
var SemanticSegmentation = /** @class */ (function () {
    function SemanticSegmentation(graphModel, base) {
        this.model = graphModel;
        this.base = base;
    }
    /**
     * Segments an arbitrary image and generates a two-dimensional tensor with
     * class labels assigned to each cell of the grid overlayed on the image ( the
     * maximum number of cells on the side is fixed to 513).
     *
     * @param input ::
     *     `ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement`
     *
     * The input image to segment.
     *
     * @return rawSegmentationMap :: `tf.Tensor2D`
     *
     * The segmentation map of the image
     */
    SemanticSegmentation.prototype.predict = function (input) {
        var _this = this;
        return tf.tidy(function () {
            var data = tf.cast(utils_1.toInputTensor(input), 'int32');
            return tf.squeeze(_this.model.execute(data));
        });
    };
    /**
     * Segments an arbitrary image and generates a two-dimensional tensor with
     * class labels assigned to each cell of the grid overlayed on the image ( the
     * maximum number of cells on the side is fixed to 513).
     *
     * @param image :: `ImageData | HTMLImageElement | HTMLCanvasElement |
     * HTMLVideoElement | tf.Tensor3D`;
     *
     *   The image to segment
     *
     * @param config (optional) The configuration object for the segmentation:
     *
     * - **config.canvas** (optional) :: `HTMLCanvasElement`
     *
     *   The canvas where to draw the output
     *
     * - **config.colormap** (optional) :: `[number, number, number][]`
     *
     *   The array of RGB colors corresponding to labels
     *
     * - **config.labels** (optional) :: `string[]`
     *
     *   The array of names corresponding to labels
     *
     *   By [default](./src/index.ts#L81), `colormap` and `labels` are set
     * according to the `base` model attribute passed during initialization.
     *
     * @returns A promise of a `DeepLabOutput` object, with four attributes:
     *
     * - **legend** :: `{ [name: string]: [number, number, number] }`
     *
     *   The legend is a dictionary of objects recognized in the image and their
     *   colors in RGB format.
     *
     * - **height** :: `number`
     *
     *   The height of the returned segmentation map
     *
     * - **width** :: `number`
     *
     *   The width of the returned segmentation map
     *
     * - **segmentationMap** :: `Uint8ClampedArray`
     *
     *   The colored segmentation map as `Uint8ClampedArray` which can be
     *   fed into `ImageData` and mapped to a canvas.
     */
    SemanticSegmentation.prototype.segment = function (input, config) {
        if (config === void 0) { config = {}; }
        return __awaiter(this, void 0, void 0, function () {
            var colormap, labels, canvas, rawSegmentationMap, _a, height, width, _b, legend, segmentationMap;
            var _this = this;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0:
                        if (!((config.colormap && config.labels) || this.base)) {
                            throw new Error("Calling the 'segment' method requires either the 'base'" +
                                " attribute to be defined " +
                                "(e.g. 'pascal', 'cityscapes' or'ade20k')," +
                                " or 'colormap' and 'labels' options to be set. " +
                                "Aborting, since neither has been provided.");
                        }
                        else if (!(config.colormap && config.labels)) {
                            config.colormap = utils_1.getColormap(this.base);
                            config.labels = utils_1.getLabels(this.base);
                        }
                        colormap = config.colormap, labels = config.labels, canvas = config.canvas;
                        rawSegmentationMap = tf.tidy(function () { return _this.predict(input); });
                        _a = rawSegmentationMap.shape, height = _a[0], width = _a[1];
                        return [4 /*yield*/, utils_1.toSegmentationImage(colormap, labels, rawSegmentationMap, canvas)];
                    case 1:
                        _b = _c.sent(), legend = _b.legend, segmentationMap = _b.segmentationMap;
                        tf.dispose(rawSegmentationMap);
                        return [2 /*return*/, { legend: legend, height: height, width: width, segmentationMap: segmentationMap }];
                }
            });
        });
    };
    /**
     * Dispose of the tensors allocated by the model.
     * You should call this when you are done with the model.
     */
    SemanticSegmentation.prototype.dispose = function () {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                if (this.model) {
                    this.model.dispose();
                }
                return [2 /*return*/];
            });
        });
    };
    return SemanticSegmentation;
}());
exports.SemanticSegmentation = SemanticSegmentation;
//# sourceMappingURL=index.js.map