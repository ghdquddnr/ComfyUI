<div align="center">

# ComfyUI
**가장 강력하고 모듈화된 시각적 AI 엔진 및 애플리케이션.**


[![Website][website-shield]][website-url]
[![Dynamic JSON Badge][discord-shield]][discord-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![][github-release-shield]][github-release-link]
[![][github-release-date-shield]][github-release-link]
[![][github-downloads-shield]][github-downloads-link]
[![][github-downloads-latest-shield]][github-downloads-link]

[matrix-shield]: https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white
[matrix-url]: https://app.element.io/#/room/%23comfyui_space%3Amatrix.org
[website-shield]: https://img.shields.io/badge/ComfyOrg-4285F4?style=flat
[website-url]: https://www.comfy.org/
<!-- Workaround to display total user from https://github.com/badges/shields/issues/4500#issuecomment-2060079995 -->
[discord-shield]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fcomfyorg%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=green&suffix=%20total
[discord-url]: https://www.comfy.org/discord

[github-release-shield]: https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver
[github-release-link]: https://github.com/comfyanonymous/ComfyUI/releases
[github-release-date-shield]: https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat
[github-downloads-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat
[github-downloads-latest-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)
</div>

ComfyUI는 그래프/노드/플로우차트 기반 인터페이스를 사용하여 고급 스테이블 디퓨전 파이프라인을 설계하고 실행할 수 있게 해줍니다. Windows, Linux, macOS에서 사용 가능합니다.

## 시작하기

#### [데스크톱 애플리케이션](https://www.comfy.org/download)
- 가장 쉽게 시작할 수 있는 방법입니다.
- Windows 및 macOS에서 사용 가능합니다.

#### [Windows 포터블 패키지](#installing)
- 최신 커밋을 얻고 완전히 휴대 가능합니다.
- Windows에서 사용 가능합니다.

#### [수동 설치](#manual-install-windows-linux)
모든 운영 체제 및 GPU 유형(NVIDIA, AMD, Intel, Apple Silicon, Ascend)을 지원합니다.

## [예제](https://comfyanonymous.github.io/ComfyUI_examples/)
[예제 워크플로우](https://comfyanonymous.github.io/ComfyUI_examples/)를 통해 ComfyUI가 할 수 있는 것들을 확인해보세요.


## 기능
- 코딩 없이 복잡한 스테이블 디퓨전 워크플로우를 실험하고 생성할 수 있는 노드/그래프/플로우차트 인터페이스
- 이미지 모델
   - SD1.x, SD2.x,
   - [SDXL](https://comfyanonymous.github.io/ComfyUI_examples/sdxl/), [SDXL Turbo](https://comfyanonymous.github.io/ComfyUI_examples/sdturbo/)
   - [Stable Cascade](https://comfyanonymous.github.io/ComfyUI_examples/stable_cascade/)
   - [SD3 및 SD3.5](https://comfyanonymous.github.io/ComfyUI_examples/sd3/)
   - Pixart Alpha 및 Sigma
   - [AuraFlow](https://comfyanonymous.github.io/ComfyUI_examples/aura_flow/)
   - [HunyuanDiT](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_dit/)
   - [Flux](https://comfyanonymous.github.io/ComfyUI_examples/flux/)
   - [Lumina Image 2.0](https://comfyanonymous.github.io/ComfyUI_examples/lumina2/)
- 비디오 모델
   - [Stable Video Diffusion](https://comfyanonymous.github.io/ComfyUI_examples/video/)
   - [Mochi](https://comfyanonymous.github.io/ComfyUI_examples/mochi/)
   - [LTX-Video](https://comfyanonymous.github.io/ComfyUI_examples/ltxv/)
   - [Hunyuan Video](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_video/)
   - [Nvidia Cosmos](https://comfyanonymous.github.io/ComfyUI_examples/cosmos/)
   - [Wan 2.1](https://comfyanonymous.github.io/ComfyUI_examples/wan/)
- 3D 모델
   - [Hunyuan3D 2.0](https://docs.comfy.org/tutorials/3d/hunyuan3D-2)
- [Stable Audio](https://comfyanonymous.github.io/ComfyUI_examples/audio/)
- 비동기 큐 시스템
- 다양한 최적화: 실행 간 변경된 워크플로우 부분만 재실행합니다.
- 스마트 메모리 관리: 1GB VRAM만 있는 GPU에서도 자동으로 모델을 실행할 수 있습니다.
- GPU가 없어도 ```--cpu``` 옵션으로 작동합니다(느림).
- ckpt, safetensors 및 diffusers 모델/체크포인트를 로드할 수 있습니다. 독립 실행형 VAE 및 CLIP 모델.
- 임베딩/텍스츄얼 인버전
- [로라(일반, locon 및 loha)](https://comfyanonymous.github.io/ComfyUI_examples/lora/)
- [하이퍼네트워크](https://comfyanonymous.github.io/ComfyUI_examples/hypernetworks/)
- 생성된 PNG, WebP 및 FLAC 파일에서 전체 워크플로우(시드 포함) 로드.
- Json 파일로 워크플로우 저장/로드.
- 노드 인터페이스로 [고해상도 수정](https://comfyanonymous.github.io/ComfyUI_examples/2_pass_txt2img/)과 같은 복잡한 워크플로우 또는 훨씬 더 고급 워크플로우를 만들 수 있습니다.
- [영역 구성](https://comfyanonymous.github.io/ComfyUI_examples/area_composition/)
- 일반 및 인페인팅 모델을 사용한 [인페인팅](https://comfyanonymous.github.io/ComfyUI_examples/inpaint/).
- [ControlNet 및 T2I-Adapter](https://comfyanonymous.github.io/ComfyUI_examples/controlnet/)
- [업스케일 모델(ESRGAN, ESRGAN 변형, SwinIR, Swin2SR 등...)](https://comfyanonymous.github.io/ComfyUI_examples/upscale_models/)
- [unCLIP 모델](https://comfyanonymous.github.io/ComfyUI_examples/unclip/)
- [GLIGEN](https://comfyanonymous.github.io/ComfyUI_examples/gligen/)
- [모델 병합](https://comfyanonymous.github.io/ComfyUI_examples/model_merging/)
- [LCM 모델 및 로라](https://comfyanonymous.github.io/ComfyUI_examples/lcm/)
- [TAESD](#how-to-show-high-quality-previews)를 이용한 잠재 공간 미리보기
- 매우 빠르게 시작됩니다.
- 완전 오프라인 작동: 어떤 것도 다운로드하지 않습니다.
- 모델 검색 경로를 설정하기 위한 [설정 파일](extra_model_paths.yaml.example).

워크플로우 예제는 [예제 페이지](https://comfyanonymous.github.io/ComfyUI_examples/)에서 찾을 수 있습니다.

## 단축키

| 단축키                            | 설명                                                                                                        |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `Ctrl` + `Enter`                      | 현재 그래프를 생성 대기열에 추가                                                                              |
| `Ctrl` + `Shift` + `Enter`              | 현재 그래프를 생성 대기열의 맨 앞에 추가                                                                     |
| `Ctrl` + `Alt` + `Enter`                | 현재 생성 취소                                                                                         |
| `Ctrl` + `Z`/`Ctrl` + `Y`                 | 실행 취소/다시 실행                                                                                                       |
| `Ctrl` + `S`                          | 워크플로우 저장                                                                                                     |
| `Ctrl` + `O`                          | 워크플로우 로드                                                                                                     |
| `Ctrl` + `A`                          | 모든 노드 선택                                                                                                  |
| `Alt `+ `C`                           | 선택한 노드 접기/펼치기                                                                                |
| `Ctrl` + `M`                          | 선택한 노드 음소거/음소거 해제                                                                                        |
| `Ctrl` + `B`                           | 선택한 노드 우회(노드가 그래프에서 제거되고 와이어가 재연결된 것처럼 작동)           |
| `Delete`/`Backspace`                   | 선택한 노드 삭제                                                                                             |
| `Ctrl` + `Backspace`                   | 현재 그래프 삭제                                                                                          |
| `Space`                              | 커서를 움직이면서 캔버스를 이동                                                            |
| `Ctrl`/`Shift` + `Click`                 | 클릭한 노드를 선택에 추가                                                                                      |
| `Ctrl` + `C`/`Ctrl` + `V`                  | 선택한 노드 복사 및 붙여넣기(선택되지 않은 노드의 출력에 대한 연결 유지 안 함)                    |
| `Ctrl` + `C`/`Ctrl` + `Shift` + `V`          | 선택한 노드 복사 및 붙여넣기(선택되지 않은 노드의 출력에서 붙여넣은 노드의 입력으로의 연결 유지) |
| `Shift` + `Drag`                       | 여러 선택한 노드를 동시에 이동                                                                     |
| `Ctrl` + `D`                           | 기본 그래프 로드                                                                                                |
| `Alt` + `+`                          | 캔버스 확대                                                                                                    |
| `Alt` + `-`                          | 캔버스 축소                                                                                                   |
| `Ctrl` + `Shift` + LMB + 수직 드래그 | 캔버스 확대/축소                                                                                               |
| `P`                                  | 선택한 노드 고정/고정 해제                                                                                          |
| `Ctrl` + `G`                           | 선택한 노드 그룹화                                                                                              |
| `Q`                                 | 대기열 가시성 전환                                                                                                    |
| `H`                                  | 히스토리 가시성 전환                                                                                      |
| `R`                                  | 그래프 새로고침                                                                                                     |
| `F`                                  | 메뉴 표시/숨기기                                                                                                     |
| `.`                                  | 선택 항목에 맞게 보기 조정(아무것도 선택되지 않았을 때는 전체 그래프)                                                       |
| 더블클릭 LMB                   | 노드 빠른 검색 팔레트 열기                                                                                    |
| `Shift` + 드래그                       | 여러 와이어를 한 번에 이동                                                                                       |
| `Ctrl` + `Alt` + LMB                   | 클릭한 슬롯에서 모든 와이어 연결 해제                                                                            |

macOS 사용자의 경우 `Ctrl` 대신 `Cmd`를 사용할 수 있습니다.

# 설치하기

## Windows 포터블

[릴리스 페이지](https://github.com/comfyanonymous/ComfyUI/releases)에는 Nvidia GPU에서 실행하거나 CPU만으로 실행할 수 있는 Windows용 포터블 독립 실행형 빌드가 있습니다.

### [다운로드 직접 링크](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)

간단히 다운로드하고 [7-Zip](https://7-zip.org)으로 압축을 풀고 실행하세요. 스테이블 디퓨전 체크포인트/모델(큰 ckpt/safetensors 파일)을 ComfyUI\models\checkpoints에 넣어야 합니다.

압축 해제에 문제가 있으면 파일을 마우스 오른쪽 버튼으로 클릭 -> 속성 -> 차단 해제

5090이나 5080과 같은 50 시리즈 Blackwell 카드를 사용하는 경우 [이 논의 스레드](https://github.com/comfyanonymous/ComfyUI/discussions/6643)를 참조하세요.

#### 다른 UI와 ComfyUI 간에 모델을 공유하려면 어떻게 해야 하나요?

모델 검색 경로를 설정하기 위한 [설정 파일](extra_model_paths.yaml.example)을 참조하세요. 독립 실행형 Windows 빌드에서는 ComfyUI 디렉토리에서 이 파일을 찾을 수 있습니다. 이 파일의 이름을 extra_model_paths.yaml로 변경하고 텍스트 편집기로 편집하세요.

## 주피터 노트북

Paperspace, Kaggle 또는 Colab과 같은 서비스에서 실행하려면 [주피터 노트북](notebooks/comfyui_colab.ipynb)을 사용할 수 있습니다.


## [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

comfy-cli를 사용하여 ComfyUI를 설치하고 시작할 수 있습니다:
```bash
pip install comfy-cli
comfy install
```

## 수동 설치 (Windows, Linux)

Python 3.13이 지원되지만 일부 사용자 정의 노드와 그 종속성이 아직 지원하지 않을 수 있으므로 3.12를 사용하는 것이 좋습니다.

이 레포지토리를 Git으로 클론하세요.

SD 체크포인트(큰 ckpt/safetensors 파일)를 models/checkpoints에 넣으세요.

VAE를 models/vae에 넣으세요.


### AMD GPU(Linux만 해당)
AMD 사용자는 이미 설치되어 있지 않은 경우 pip를 사용하여 rocm과 pytorch를 설치할 수 있습니다. 안정 버전을 설치하는 명령은 다음과 같습니다:

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4```

일부 성능 향상이 있을 수 있는 ROCm 6.3이 포함된 나이틀리를 설치하는 명령은 다음과 같습니다:

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3```

### Intel GPU(Windows 및 Linux)

(옵션 1) Intel Arc GPU 사용자는 pip를 사용하여 torch.xpu 지원이 포함된 네이티브 PyTorch를 설치할 수 있습니다(현재 PyTorch 나이틀리 빌드에서 사용 가능). 자세한 정보는 [여기](https://pytorch.org/docs/main/notes/get_start_xpu.html)에서 찾을 수 있습니다.
  
1. PyTorch 나이틀리를 설치하려면 다음 명령을 사용하세요:

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu```

2. `python main.py`를 실행하여 ComfyUI를 시작하세요.


(옵션 2) 또는 Intel Extension for PyTorch(IPEX)가 지원하는 Intel GPU는 성능 향상을 위해 IPEX를 활용할 수 있습니다.

1. IPEX를 활용하는 Intel® Arc™ A-시리즈 그래픽의 경우, conda 환경을 만들고 아래 명령을 사용하세요:

```
conda install libuv
pip install torch==2.3.1.post0+cxx11.abi torchvision==0.18.1.post0+cxx11.abi torchaudio==2.3.1.post0+cxx11.abi intel-extension-for-pytorch==2.3.110.post0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
```

IPEX가 있는 다른 지원되는 Intel GPU의 경우, 자세한 정보는 [설치](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu)를 방문하세요.

추가 논의 및 도움은 [여기](https://github.com/comfyanonymous/ComfyUI/discussions/476)에서 찾을 수 있습니다.

### NVIDIA

Nvidia 사용자는 이 명령을 사용하여 안정 pytorch를 설치해야 합니다:

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126```

이것은 새로운 blackwell 50xx 시리즈 GPU를 지원하고 성능 향상이 있을 수 있는 pytorch 나이틀리를 대신 설치하는 명령입니다.

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128```

#### 문제 해결

"Torch not compiled with CUDA enabled" 오류가 발생하면 다음으로 torch를 제거하세요:

```pip uninstall torch```

그리고 위의 명령으로 다시 설치하세요.

### 종속성

ComfyUI 폴더 내에서 터미널을 열고 다음을 실행하여 종속성을 설치하세요:

```pip install -r requirements.txt```

이제 모든 것이 설치되었으므로 ComfyUI를 실행할 수 있습니다.

### 기타:

#### Apple Mac 실리콘

최신 macOS 버전으로 Apple Mac 실리콘(M1 또는 M2)에 ComfyUI를 설치할 수 있습니다.

1. PyTorch 나이틀리를 설치하세요. 지침은 Apple 개발자 가이드 [Mac에서 가속화된 PyTorch 훈련](https://developer.apple.com/metal/pytorch/)을 읽으세요(최신 PyTorch 나이틀리를 설치해야 합니다).
1. Windows 및 Linux용 [ComfyUI 수동 설치](#manual-install-windows-linux) 지침을 따르세요.
1. ComfyUI [종속성](#dependencies)을 설치하세요. 다른 스테이블 디퓨전 UI가 있는 경우 [종속성을 재사용](#i-already-have-another-ui-for-stable-diffusion-installed-do-i-really-have-to-install-all-of-these-dependencies)할 수 있습니다.
1. `python main.py`를 실행하여 ComfyUI를 시작하세요.

> **참고**: [ComfyUI 수동 설치](#manual-install-windows-linux)에서 설명한 대로 모델, VAE, LoRA 등을 해당 Comfy 폴더에 추가하는 것을 잊지 마세요.

#### DirectML(Windows의 AMD 카드)

```pip install torch-directml```을 실행한 다음 다음 명령으로 ComfyUI를 시작할 수 있습니다: ```python main.py --directml```

#### Ascend NPU

Ascend Extension for PyTorch(torch_npu)와 호환되는 모델용입니다. 시작하려면 [설치](https://ascend.github.io/docs/sources/ascend/quick_install.html) 페이지에 설명된 사전 요구 사항을 환경이 충족하는지 확인하세요. 다음은 플랫폼 및 설치 방법에 맞게 조정된 단계별 가이드입니다:

1. 필요한 경우 torch-npu의 설치 페이지에 지정된 대로 Linux의 권장 커널 버전 또는 최신 버전을 설치하세요.
2. 플랫폼별 지침에 따라 드라이버, 펌웨어 및 CANN이 포함된 Ascend Basekit 설치를 진행하세요.
3. 다음으로 [설치](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch) 페이지의 플랫폼별 지침을 따라 torch-npu에 필요한 패키지를 설치하세요.
4. 마지막으로 Linux용 [ComfyUI 수동 설치](#manual-install-windows-linux) 가이드를 따르세요. 모든 구성 요소가 설치되면 이전에 설명한 대로 ComfyUI를 실행할 수 있습니다.

#### Cambricon MLU

Cambricon Extension for PyTorch(torch_mlu)와 호환되는 모델용입니다. 다음은 플랫폼 및 설치 방법에 맞게 조정된 단계별 가이드입니다:

1. [설치](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html)의 플랫폼별 지침을 따라 Cambricon CNToolkit을 설치하세요.
2. 다음으로 [설치](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html) 지침에 따라 PyTorch(torch_mlu)를 설치하세요.
3. `python main.py`를 실행하여 ComfyUI를 시작하세요.

# 실행하기

```python main.py```

### ROCm에서 공식적으로 지원하지 않는 AMD 카드의 경우

문제가 있는 경우 다음 명령으로 실행해 보세요:

6700, 6600 및 기타 RDNA2 또는 이전 버전: ```HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py```

AMD 7600 및 기타 RDNA3 카드: ```HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py```

### AMD ROCm 팁

다음 명령을 사용하여 RDNA3 및 기타 잠재적 AMD GPU에서 pytorch 2.5의 ComfyUI에서 실험적 메모리 효율적 주의를 활성화할 수 있습니다:

```TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention```

또한 이 환경 변수 `PYTORCH_TUNABLEOP_ENABLED=1`를 설정해 볼 수 있습니다. 초기 실행이 매우 느려지는 대신 속도가 빨라질 수 있습니다.

# 참고

올바른 입력이 있는 출력이 있는 그래프의 일부만 실행됩니다.

각 실행마다 변경되는 그래프의 일부만 실행됩니다. 동일한 그래프를 두 번 제출하면 첫 번째만 실행됩니다. 그래프의 마지막 부분을 변경하면 변경한 부분과 그에 의존하는 부분만 실행됩니다.

웹페이지에 생성된 png를 드래그하거나 로드하면 시드를 포함한 전체 워크플로우를 볼 수 있습니다.

(good code:1.2) 또는 (bad code:0.8)과 같이 () 를 사용하여 단어나 구문의 강조를 변경할 수 있습니다. ()의 기본 강조는 1.1입니다. 실제 프롬프트에서 () 문자를 사용하려면 \\( 또는 \\)와 같이 이스케이프하세요.

동적 프롬프트에 {day|night}와 같은 와일드카드를 사용할 수 있습니다. 이 구문 "{wild|card|test}"는 프롬프트를 대기열에 넣을 때마다 프론트엔드에서 무작위로 "wild", "card" 또는 "test" 중 하나로 대체됩니다. 실제 프롬프트에서 {} 문자를 사용하려면 \\{ 또는 \\}와 같이 이스케이프하세요.

동적 프롬프트는 또한 `// 주석` 또는 `/* 주석 */`과 같은 C 스타일 주석을 지원합니다.

텍스트 프롬프트에서 텍스트 반전 개념/임베딩을 사용하려면 models/embeddings 디렉토리에 넣고 다음과 같이 CLIPTextEncode 노드에서 사용하세요(.pt 확장자는 생략할 수 있습니다):

```embedding:embedding_filename.pt```


## 고품질 미리보기를 표시하려면 어떻게 해야 하나요?

미리보기를 활성화하려면 ```--preview-method auto```를 사용하세요.

기본 설치에는 저해상도의 빠른 잠재 공간 미리보기 방법이 포함되어 있습니다. [TAESD](https://github.com/madebyollin/taesd)를 사용하여 고품질 미리보기를 활성화하려면 [taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth 및 taef1_decoder.pth](https://github.com/madebyollin/taesd/)를 다운로드하여 `models/vae_approx` 폴더에 넣으세요. 설치 후 ComfyUI를 다시 시작하고 `--preview-method taesd`로 실행하여 고품질 미리보기를 활성화하세요.

## TLS/SSL을 사용하려면 어떻게 해야 하나요?
자체 서명 인증서(공유/프로덕션 용도에는 적합하지 않음)와 키를 생성하려면 다음 명령을 실행하세요: `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`

TLS/SSL을 활성화하려면 `--tls-keyfile key.pem --tls-certfile cert.pem`을 사용하세요. 이제 앱은 `http://...` 대신 `https://...`로 접근할 수 있습니다.

> 참고: Windows 사용자는 [alexisrolland/docker-openssl](https://github.com/alexisrolland/docker-openssl) 또는 [제3자 바이너리 배포판](https://wiki.openssl.org/index.php/Binaries) 중 하나를 사용하여 위 명령 예제를 실행할 수 있습니다.
<br/><br/>컨테이너를 사용하는 경우 볼륨 마운트 `-v`는 상대 경로일 수 있으므로 `... -v ".\:/openssl-certs" ...`는 명령 프롬프트 또는 PowerShell 터미널의 현재 디렉토리에 키 및 인증서 파일을 생성합니다.

## 지원 및 개발 채널

[Discord](https://comfy.org/discord): #help 또는 #feedback 채널을 시도해 보세요.

[Matrix 스페이스: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org) (Discord와 비슷하지만 오픈 소스입니다).

또한 참조: [https://www.comfy.org/](https://www.comfy.org/)

## 프론트엔드 개발

2024년 8월 15일부터 새로운 프론트엔드로 전환했으며, 이제 별도의 저장소에 호스팅됩니다: [ComfyUI 프론트엔드](https://github.com/Comfy-Org/ComfyUI_frontend). 이 저장소는 이제 `web/` 디렉토리 아래에 컴파일된 JS(TS/Vue에서)를 호스팅합니다.

### 문제 보고 및 기능 요청

프론트엔드와 관련된 버그, 문제 또는 기능 요청은 [ComfyUI 프론트엔드 저장소](https://github.com/Comfy-Org/ComfyUI_frontend)를 사용하세요. 이를 통해 프론트엔드 관련 문제를 보다 효율적으로 관리하고 해결할 수 있습니다.

### 최신 프론트엔드 사용하기

새로운 프론트엔드는 이제 ComfyUI의 기본입니다. 그러나 다음 사항에 유의하세요:

1. 메인 ComfyUI 저장소의 프론트엔드는 2주마다 업데이트됩니다.
2. 일일 릴리스는 별도의 프론트엔드 저장소에서 사용할 수 있습니다.

최신 프론트엔드 버전을 사용하려면:

1. 최신 일일 릴리스의 경우 다음 명령줄 인수로 ComfyUI를 시작하세요:

   ```
   --front-end-version Comfy-Org/ComfyUI_frontend@latest
   ```

2. 특정 버전의 경우 `latest`를 원하는 버전 번호로 바꾸세요:

   ```
   --front-end-version Comfy-Org/ComfyUI_frontend@1.2.2
   ```

이 접근 방식을 통해 안정적인 2주 릴리스와 최신 일일 업데이트 사이를 쉽게 전환하거나 테스트 목적으로 특정 버전을 사용할 수 있습니다.

### 레거시 프론트엔드 접근하기

어떤 이유로든 레거시 프론트엔드를 사용해야 하는 경우 다음 명령줄 인수를 사용하여 접근할 수 있습니다:

```
--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest
```

이것은 [ComfyUI 레거시 프론트엔드 저장소](https://github.com/Comfy-Org/ComfyUI_legacy_frontend)에 보존된 레거시 프론트엔드의 스냅샷을 사용합니다.

# QA

### 이를 위해 어떤 GPU를 구매해야 하나요?

[일부 추천 사항은 이 페이지를 참조하세요](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)
