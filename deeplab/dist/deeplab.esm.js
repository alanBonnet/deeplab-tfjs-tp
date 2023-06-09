/**
    * @license
    * Copyright 2023 Google LLC. All Rights Reserved.
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
/**
    * @license
    * Copyright 2023 Google LLC. All Rights Reserved.
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
import{loadGraphModel as e}from"@tensorflow/tfjs-converter";import*as tf from"@tensorflow/tfjs-core";import{tidy as t,Tensor as n,browser as r,expandDims as o,image as a,buffer as i,dispose as s,cast as l,squeeze as c}from"@tensorflow/tfjs-core";
/*! *****************************************************************************
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

THIS CODE IS PROVIDED ON AN *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
MERCHANTABLITY OR NON-INFRINGEMENT.

See the Apache Version 2.0 License for specific language governing permissions
and limitations under the License.
***************************************************************************** */function u(e,t,n,r){return new(n||(n=Promise))((function(o,a){function i(e){try{l(r.next(e))}catch(e){a(e)}}function s(e){try{l(r.throw(e))}catch(e){a(e)}}function l(e){e.done?o(e.value):new n((function(t){t(e.value)})).then(i,s)}l((r=r.apply(e,t||[])).next())}))}function f(e,t){var n,r,o,a,i={label:0,sent:function(){if(1&o[0])throw o[1];return o[1]},trys:[],ops:[]};return a={next:s(0),throw:s(1),return:s(2)},"function"==typeof Symbol&&(a[Symbol.iterator]=function(){return this}),a;function s(a){return function(s){return function(a){if(n)throw new TypeError("Generator is already executing.");for(;i;)try{if(n=1,r&&(o=2&a[0]?r.return:a[0]?r.throw||((o=r.return)&&o.call(r),0):r.next)&&!(o=o.call(r,a[1])).done)return o;switch(r=0,o&&(a=[2&a[0],o.value]),a[0]){case 0:case 1:o=a;break;case 4:return i.label++,{value:a[1],done:!1};case 5:i.label++,r=a[1],a=[0];continue;case 7:a=i.ops.pop(),i.trys.pop();continue;default:if(!(o=i.trys,(o=o.length>0&&o[o.length-1])||6!==a[0]&&2!==a[0])){i=0;continue}if(3===a[0]&&(!o||a[1]>o[0]&&a[1]<o[3])){i.label=a[1];break}if(6===a[0]&&i.label<o[1]){i.label=o[1],o=a;break}if(o&&i.label<o[2]){i.label=o[2],i.ops.push(a);break}o[2]&&i.ops.pop(),i.trys.pop();continue}a=t.call(e,i)}catch(e){a=[6,e],r=0}finally{n=o=0}if(5&a[0])throw a[1];return{value:a[0]?a[1]:void 0,done:!0}}([a,s])}}}
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
 */var d="https://tfhub.dev/tensorflow/tfjs-model/deeplab",h=513,p={ADE20K:[[0,0,0],[120,120,120],[180,120,120],[6,230,230],[80,50,50],[4,200,3],[120,120,80],[140,140,140],[204,5,255],[230,230,230],[4,250,7],[224,5,255],[235,255,7],[150,5,61],[120,120,70],[8,255,51],[255,6,82],[143,255,140],[204,255,4],[255,51,7],[204,70,3],[0,102,200],[61,230,250],[255,6,51],[11,102,255],[255,7,71],[255,9,224],[9,7,230],[220,220,220],[255,9,92],[112,9,255],[8,255,214],[7,255,224],[255,184,6],[10,255,71],[255,41,10],[7,255,255],[224,255,8],[102,8,255],[255,61,6],[255,194,7],[255,122,8],[0,255,20],[255,8,41],[255,5,153],[6,51,255],[235,12,255],[160,150,20],[0,163,255],[140,140,140],[250,10,15],[20,255,0],[31,255,0],[255,31,0],[255,224,0],[153,255,0],[0,0,255],[255,71,0],[0,235,255],[0,173,255],[31,0,255],[11,200,200],[255,82,0],[0,255,245],[0,61,255],[0,255,112],[0,255,133],[255,0,0],[255,163,0],[255,102,0],[194,255,0],[0,143,255],[51,255,0],[0,82,255],[0,255,41],[0,255,173],[10,0,255],[173,255,0],[0,255,153],[255,92,0],[255,0,255],[255,0,245],[255,0,102],[255,173,0],[255,0,20],[255,184,184],[0,31,255],[0,255,61],[0,71,255],[255,0,204],[0,255,194],[0,255,82],[0,10,255],[0,112,255],[51,0,255],[0,194,255],[0,122,255],[0,255,163],[255,153,0],[0,255,10],[255,112,0],[143,255,0],[82,0,255],[163,255,0],[255,235,0],[8,184,170],[133,0,255],[0,255,92],[184,0,255],[255,0,31],[0,184,255],[0,214,255],[255,0,112],[92,255,0],[0,224,255],[112,224,255],[70,184,160],[163,0,255],[153,0,255],[71,255,0],[255,0,163],[255,204,0],[255,0,143],[0,255,235],[133,255,0],[255,0,235],[245,0,255],[255,0,122],[255,245,0],[10,190,212],[214,255,0],[0,204,255],[20,0,255],[255,255,0],[0,153,255],[0,41,255],[0,255,204],[41,0,255],[41,255,0],[173,0,255],[0,245,255],[71,0,255],[122,0,255],[0,255,184],[0,92,255],[184,255,0],[0,133,255],[255,214,0],[25,194,194],[102,255,0],[92,0,255]],CITYSCAPES:[[128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],[70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],[0,80,100],[0,0,230],[119,11,32]],PASCAL:[[0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],[64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],[192,128,128],[0,64,0],[128,64,0],[0,192,0],[128,192,0],[0,64,128],[128,64,128],[0,192,128],[128,192,128],[64,64,0],[192,64,0],[64,192,0],[192,192,0],[64,64,128],[192,64,128],[64,192,128],[192,192,128],[0,0,64],[128,0,64],[0,128,64],[128,128,64],[0,0,192],[128,0,192],[0,128,192],[128,128,192],[64,0,64],[192,0,64],[64,128,64],[192,128,64],[64,0,192],[192,0,192],[64,128,192],[192,128,192],[0,64,64],[128,64,64],[0,192,64],[128,192,64],[0,64,192],[128,64,192],[0,192,192],[128,192,192],[64,64,64],[192,64,64],[64,192,64],[192,192,64],[64,64,192],[192,64,192],[64,192,192],[192,192,192],[32,0,0],[160,0,0],[32,128,0],[160,128,0],[32,0,128],[160,0,128],[32,128,128],[160,128,128],[96,0,0],[224,0,0],[96,128,0],[224,128,0],[96,0,128],[224,0,128],[96,128,128],[224,128,128],[32,64,0],[160,64,0],[32,192,0],[160,192,0],[32,64,128],[160,64,128],[32,192,128],[160,192,128],[96,64,0],[224,64,0],[96,192,0],[224,192,0],[96,64,128],[224,64,128],[96,192,128],[224,192,128],[32,0,64],[160,0,64],[32,128,64],[160,128,64],[32,0,192],[160,0,192],[32,128,192],[160,128,192],[96,0,64],[224,0,64],[96,128,64],[224,128,64],[96,0,192],[224,0,192],[96,128,192],[224,128,192],[32,64,64],[160,64,64],[32,192,64],[160,192,64],[32,64,192],[160,64,192],[32,192,192],[160,192,192],[96,64,64],[224,64,64],[96,192,64],[224,192,64],[96,64,192],[224,64,192],[96,192,192],[224,192,192],[0,32,0],[128,32,0],[0,160,0],[128,160,0],[0,32,128],[128,32,128],[0,160,128],[128,160,128],[64,32,0],[192,32,0],[64,160,0],[192,160,0],[64,32,128],[192,32,128],[64,160,128],[192,160,128],[0,96,0],[128,96,0],[0,224,0],[128,224,0],[0,96,128],[128,96,128],[0,224,128],[128,224,128],[64,96,0],[192,96,0],[64,224,0],[192,224,0],[64,96,128],[192,96,128],[64,224,128],[192,224,128],[0,32,64],[128,32,64],[0,160,64],[128,160,64],[0,32,192],[128,32,192],[0,160,192],[128,160,192],[64,32,64],[192,32,64],[64,160,64],[192,160,64],[64,32,192],[192,32,192],[64,160,192],[192,160,192],[0,96,64],[128,96,64],[0,224,64],[128,224,64],[0,96,192],[128,96,192],[0,224,192],[128,224,192],[64,96,64],[192,96,64],[64,224,64],[192,224,64],[64,96,192],[192,96,192],[64,224,192],[192,224,192],[32,32,0],[160,32,0],[32,160,0],[160,160,0],[32,32,128],[160,32,128],[32,160,128],[160,160,128],[96,32,0],[224,32,0],[96,160,0],[224,160,0],[96,32,128],[224,32,128],[96,160,128],[224,160,128],[32,96,0],[160,96,0],[32,224,0],[160,224,0],[32,96,128],[160,96,128],[32,224,128],[160,224,128],[96,96,0],[224,96,0],[96,224,0],[224,224,0],[96,96,128],[224,96,128],[96,224,128],[224,224,128],[32,32,64],[160,32,64],[32,160,64],[160,160,64],[32,32,192],[160,32,192],[32,160,192],[160,160,192],[96,32,64],[224,32,64],[96,160,64],[224,160,64],[96,32,192],[224,32,192],[96,160,192],[224,160,192],[32,96,64],[160,96,64],[32,224,64],[160,224,64],[32,96,192],[160,96,192],[32,224,192],[160,224,192],[96,96,64],[224,96,64],[96,224,64],[224,224,64],[96,96,192],[224,96,192],[96,224,192],[224,224,192]]},b={PASCAL:["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","dining table","dog","horse","motorbike","person","potted plant","sheep","sofa","train","TV"],CITYSCAPES:["road","sidewalk","building","wall","fence","pole","traffic light","traffic sign","vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"],ADE20K:["background","wall","building","sky","floor","tree","ceiling","road","bed","windowpane","grass","cabinet","sidewalk","person","earth","door","table","mountain","plant","curtain","chair","car","water","painting","sofa","shelf","house","sea","mirror","rug","field","armchair","seat","fence","desk","rock","wardrobe","lamp","bathtub","railing","cushion","base","box","column","signboard","chest","counter","sand","sink","skyscraper","fireplace","refrigerator","grandstand","path","stairs","runway","case","pool","pillow","screen","stairway","river","bridge","bookcase","blind","coffee","toilet","flower","book","hill","bench","countertop","stove","palm","kitchen","computer","swivel","boat","bar","arcade","hovel","bus","towel","light","truck","tower","chandelier","awning","streetlight","booth","television","airplane","dirt","apparel","pole","land","bannister","escalator","ottoman","bottle","buffet","poster","stage","van","ship","fountain","conveyer","canopy","washer","plaything","swimming","stool","barrel","basket","waterfall","tent","bag","minibike","cradle","oven","ball","food","step","tank","trade","microwave","pot","animal","bicycle","lake","dishwasher","screen","blanket","sculpture","hood","sconce","vase","traffic","tray","ashcan","fan","pier","screen","plate","monitor","bulletin","shower","radiator","glass","clock","flag"]};
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
 */function m(e,t){return""+d+"/"+(4===t?e+"/1/default/1/model.json":e+"/1/quantized/"+t+"/1/model.json")+"?tfjs-format=file"}function w(e){if("pascal"===e)return p.PASCAL;if("ade20k"===e)return p.ADE20K;if("cityscapes"===e)return p.CITYSCAPES;throw new Error("SemanticSegmentation cannot be constructed with an invalid base model "+e+". Try one of 'pascal', 'cityscapes' and 'ade20k'.")}function g(e){if("pascal"===e)return b.PASCAL;if("ade20k"===e)return b.ADE20K;if("cityscapes"===e)return b.CITYSCAPES;throw new Error("SemanticSegmentation cannot be constructed with an invalid base model "+e+". Try one of 'pascal', 'cityscapes' and 'ade20k'.")}function y(e,t,n,o){return u(this,void 0,void 0,(function(){var a,l,c,u,d,h,p,b,m,w,g,y,v,k;return f(this,(function(f){switch(f.label){case 0:if(e.length<t.length)throw new Error("The colormap must be expansive enough to encode each label. Aborting, since the given colormap has length "+e.length+", but there are "+t.length+" labels.");return a=n.shape,l=a[0],c=a[1],u=i([l,c,3],"int32"),[4,n.array()];case 1:for(d=f.sent(),h=new Set,p=0;p<l;++p)for(b=0;b<c;++b)k=d[p][b],h.add(k),u.set(e[k][0],p,b,0),u.set(e[k][1],p,b,1),u.set(e[k][2],p,b,2);return m=u.toTensor(),[4,r.toPixels(m,o)];case 2:for(w=f.sent(),s(m),g={},y=0,v=Array.from(h);y<v.length;y++)k=v[y],g[t[k]]=e[k];return[2,{legend:g,segmentationMap:w}]}}))}))}
/** @license See the LICENSE file. */var v="0.2.1";
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
 */function k(t){return void 0===t&&(t={base:"pascal",quantizationBytes:2}),u(this,void 0,void 0,(function(){var n;return f(this,(function(r){switch(r.label){case 0:if(null==tf)throw new Error("Cannot find TensorFlow.js. If you are using a <script> tag, please also include @tensorflow/tfjs on the page before using this model.");if(t.base){if(-1===["pascal","cityscapes","ade20k"].indexOf(t.base))throw new Error("SemanticSegmentation cannot be constructed with an invalid base model "+t.base+". Try one of 'pascal', 'cityscapes' and 'ade20k'.");if(-1===[1,2,4].indexOf(t.quantizationBytes))throw new Error("Only quantization to 1, 2 or 4 bytes is supported.")}else if(!t.modelUrl)throw new Error("SemanticSegmentation can be constructed either by passing the weights URL or one of the supported base model names from 'pascal', 'cityscapes' and 'ade20k',together with the degree of quantization (either 1, 2 or 4).Aborting, since neither has been provided.");return[4,e(t.modelUrl||m(t.base,t.quantizationBytes))];case 1:return n=r.sent(),[2,new S(n,t.base)]}}))}))}var S=function(){function e(e,t){this.model=e,this.base=t}return e.prototype.predict=function(e){var i=this;return t((function(){var s=l(function(e){return t((function(){var t=e instanceof n?e:r.fromPixels(e),i=t.shape,s=i[0],l=i[1],c=h/Math.max(l,s),u=Math.round(s*c),f=Math.round(l*c);return o(a.resizeBilinear(t,[u,f]))}))}(e),"int32");return c(i.model.execute(s))}))},e.prototype.segment=function(e,n){return void 0===n&&(n={}),u(this,void 0,void 0,(function(){var r,o,a,i,l,c,u,d,h,p,b=this;return f(this,(function(f){switch(f.label){case 0:if(!(n.colormap&&n.labels||this.base))throw new Error("Calling the 'segment' method requires either the 'base' attribute to be defined (e.g. 'pascal', 'cityscapes' or'ade20k'), or 'colormap' and 'labels' options to be set. Aborting, since neither has been provided.");return n.colormap&&n.labels||(n.colormap=w(this.base),n.labels=g(this.base)),r=n.colormap,o=n.labels,a=n.canvas,i=t((function(){return b.predict(e)})),l=i.shape,c=l[0],u=l[1],[4,y(r,o,i,a)];case 1:return d=f.sent(),h=d.legend,p=d.segmentationMap,s(i),[2,{legend:h,height:c,width:u,segmentationMap:p}]}}))}))},e.prototype.dispose=function(){return u(this,void 0,void 0,(function(){return f(this,(function(e){return this.model&&this.model.dispose(),[2]}))}))},e}();export{S as SemanticSegmentation,w as getColormap,g as getLabels,m as getURL,k as load,y as toSegmentationImage,v as version};
