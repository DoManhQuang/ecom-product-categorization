# MEPC: Multi-level Product Category Recognition Image Dataset

## Summary
![img3](https://huggingface.co/datasets/sherlockvn/MEPC/resolve/main/assets/posterOfMEPC.jpg)

## Word Cloud
![img](https://huggingface.co/datasets/sherlockvn/MEPC/resolve/main/assets/wordcloud-1.png)

## Introduce
### MEPC - 1000 Dataset:
- Classes: 1000
- Images: 164,117
- Train: 131,293
- Val: 32824
### MEPC - 10 Dataset:
- Classes: 10
- Images: 2,192
- Train: 1,753
- Val: 439

#### Fig: Label-only embeddings visualizing label connections of the MEPC dataset, with multi-colored labels for level 1, yellow for level 2, gray for level 3, red connections from level 1 to level 2, and blue connections from level 2 to level 3

![img2](https://huggingface.co/datasets/sherlockvn/MEPC/resolve/main/assets/netwokrx-1.png)

## Benchmark

#### Table 1: Benchmark K-fold cho MEPC-10 Dataset (K=5)
| Name          | MEPC-10<sup>Top-1</sup> (%)      | #Params  |
|---------------|----------------------------------|----------|
| MoBiNet  | 79.041% ± 2.978                 | 4.3M     |
| MoBiNetV2| 72.694% ± 1.766                 | 3.5M     |
| MoBiNetV3-S| 89.224% ± 1.185               | 2.5M     |
| MoBiNetV3-L| 92.055% ± 0.602               | 5.4M     |
| ResNet50| 89.817% ± 0.897                 | 25.6M    |
| VGG16 | 90.457% ± 1.203                 | 138.4M   |
| EfficientNetB4 | 91.37% ± 1.68              | 19.5M    |


#### Table 2: Experimental results with YOLOv8 Model
| Name       | YOLOv8m (17M params)          |              |
|------------|-------------------------------|--------------|
|            | Top-1 Accuracy (%)            | Top-5 Accuracy (%) |
| MEPC-10    | 90.87%                        | -            |
| MEPC-1000  | 34.41%                        | 57.36%       |


```s
THE DATA IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE DATA OR THE USE OR OTHER DEALINGS IN THE DATA.
```