
## DETR

DETR은 Object Detection의 SOTA모델에 대부분에 적용되는 아키텍쳐이다.

Object Detection을 넘어서 위 레포에 d2(=detectron2)를 Wrapping 해놓았다. <br/>
이를 통해서 Panoptic Segmentation을 진행할 수 있다.


### 수행한 작업
* 레포에 있는, hands_on, detection, panoptic-segmentation standalone colab을 돌려보면서 DETR 아키텍쳐 동작 원리 파악
  * 여기서는 모델을 simplify해서 pre-trained된 모델에 대해 이미지를 검증해보는 작업을 시도해볼 수 있었음
  * 그래고 attention-map을 시각화 하는 작업을 통해서 encoder-decoder에서의 abliation study를 하며 이들을 가시화 할 수 있었음
* 실제 논문 환경으로 구축된 아키텍쳐에 학습을 하려면 Transfer-Running을 해야하지만 Issue에서 이가 잘 동작하는지에 대해서는 검증하지 못했다고 되어있음
  * 그래도 아마 잘 동작될 것으로 예상해서, (우선 Detection 먼저) Custom face dataset을 coco format으로 바꿈
* 이제 해당 face dataset을 git 레포에 있던 DETR-DC5 model을 통해 train 가중치를 초기화 시켜주었음
  * 하지만 논문에서도 나와있었지만, 1epoch에 40분정도가 걸려 최소 200은 넘게 돌려야한다는데, 이를 할 컴퓨팅 파워가 되지 못하였음
  * 심지어 mac 환경에서 돌려야 했는데, gpu가 있는 서버가 없어서 이를 1batch만을 학습하는데에도 10분이 걸렸음,,, (불가능)
  * 아마 이렇게 전이학습을 통해 모델을 fine-tunning하게 되면, bounding box를 잘 찾을 것으로 예상.
  * 그리고 실제로 코드의 많은 부분을 수정해야 하는데, 이는 datasets/face.py에 수정한 버전을 올려두었음
      * face를 fine-tunning할떄에는 class가 얼굴 1개여서 이에 대한 detr.py도 수정해야하고,, 텐서 차원도 수정해야 하고,, --num_queries=100(=default)에서 좀 줄여주어야 함(10?)
* 그래서 그냥 pre-trained된 모델에 위에 hands-on 파일을 참조하여 detection 결과와 attention-matrix들을 encoder, decoder에서 찍어보는 작업만 일단 진행함.
  * 잘 동작하는듯
* panoptic-segmentation도 시간이 부족해서 해보진 않았지만, 아마 Detection이 잘 동작하니까 Detectron2를 활용하면 colab에서 봤던것과 같이 잘 돌아갈듯.


## .vscode/launch.json
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "main.py",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": ["--batch_size", "2",
                "--no_aux_loss",
                "--eval",
                "--device",
                "cpu", 
                "--dataset_file", "face", 
                "--data_path", "./workspace/custom/dataset/", 
                "--resume", "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"]
        },
        {
            "name": "test.py",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": [
                "--device", "cpu",
                "--data_path", "./workspace/custom/dataset/coco_test/",
                "--dataset_file", "coco",
                "--resume", "weights/detr-r50-e632da11.pth"
            ]
        }
    ]
}
```


## COCO 2017 valk5 Detection Result

![Screenshot 2023-07-28 at 3 19 34 PM](https://github.com/eunoiahyunseo/detr-study/assets/59719629/12cc9dc1-93dd-45ee-a77d-e52c4284b811)




## Encoder-Decoder MHA Attention-Map Check

![Screenshot 2023-07-28 at 3 42 00 PM](https://github.com/eunoiahyunseo/detr-study/assets/59719629/74865a8f-5183-4cad-9ea9-cfe4d650749c)

![Screenshot 2023-07-28 at 3 43 00 PM](https://github.com/eunoiahyunseo/detr-study/assets/59719629/ead35d41-b394-4714-a47a-d43143084a5e)


