# 手势控制3D模型旋转 - 技术研究报告

## 1. 项目概述

本项目旨在通过摄像头捕捉用户手势，将手势动作映射为3D模型的旋转控制。用户可以通过滑动手指来旋转3D模型，实现自然、直观的交互体验。

## 2. 技术方案分析

### 2.1 手势检测技术选型

#### 2.1.1 MediaPipe Hands (推荐)

Google开发的跨平台机器学习框架，提供高精度的手部21个关键点3D坐标检测。

**核心特性：**
- 检测21个3D手部关键点（21 landmarks）
- 支持双手检测
- 实时性能：30+ FPS
- 模型复杂度可选（0, 1, 2）
- 支持左右手识别

**性能指标：**
| 配置 | 检测精度 | 追踪精度 | 性能 |
|------|---------|---------|------|
| complexity=0 | 0.5 | 0.5 | 最快 |
| complexity=1 | 0.7 | 0.7 | 中等 |
| complexity=2 | 0.9 | 0.9 | 最慢 |

**优点：**
- 预训练模型，无需训练数据
- 跨平台支持（Windows, macOS, Linux, 移动端）
- 官方提供Python API
- 社区活跃，文档完善

**缺点：**
- 需要安装mediapipe依赖
- 在低端设备上可能掉帧

#### 2.1.2 替代方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| MediaPipe Hands | 精度高，开箱即用 | Google生态 | 推荐首选 |
| OpenCV + DNN | 灵活定制 | 需要自己训练模型 | 特殊需求 |
| TensorFlow Lite | 移动端优化 | 精度一般 | 嵌入式场景 |

### 2.2 手势识别方案

#### 2.2.1 静态手势识别
识别特定手势（如拳头、剪刀、石头等），适用于触发特定动作。

#### 2.2.2 动态手势追踪（推荐用于旋转控制）
追踪手部关键点的连续运动轨迹，用于控制连续动作：
- **指尖追踪**：追踪食指或中指指尖的位置变化
- **手掌方向**：检测手掌的俯仰角(Pitch)和翻转角(Roll)
- **滑动检测**：检测水平/垂直滑动方向和速度

**滑动检测算法：**
```
1. 记录上一帧指尖位置 (prev_x, prev_y)
2. 记录当前帧指尖位置 (curr_x, curr_y)
3. 计算位移向量 (dx, curr_x - prev_x, dy = curr_y - prev_y)
4. 判断滑动方向：
   - |dx| > |dy| → 水平滑动
   - |dy| > |dx| → 垂直滑动
5. 根据位移量计算旋转速度
```

### 2.3 3D渲染技术选型

#### 2.3.1 Pygame + PyOpenGL (推荐)

**技术栈：**
- **Pygame**：窗口管理、事件处理
- **PyOpenGL**：OpenGL绑定，3D渲染

**优点：**
- 纯Python实现，依赖少
- 跨平台支持
- 性能足够
- 社区成熟

**示例依赖：**
```toml
pygame>=2.0
PyOpenGL>=3.1
numpy>=1.20
```

#### 2.3.2 其他3D方案对比

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|-------|
| Pygame + PyOpenGL | 简单，依赖少 | 需要自己实现很多功能 | ★★★★★ |
| ModernGL | 现代OpenGL，更快 | 文档较少 | ★★★★ |
| VisPy | 科学可视化 | 不适合游戏化渲染 | ★★★ |
| Three.js + Flask | 视觉效果好 | 需要前后端分离 | ★★★★ |
| Ursina | 上手简单 | 功能有限 | ★★★★ |

### 2.4 手势到旋转的映射方案

#### 2.4.1 方案一：滑动方向控制（简单）

```python
# 水平滑动 → Y轴旋转
# 垂直滑动 → X轴旋转

def map_gesture_to_rotation(dx, dy):
    rotation_x = dy * sensitivity  # 垂直滑动控制X轴
    rotation_y = dx * sensitivity  # 水平滑动控制Y轴
    return rotation_x, rotation_y
```

#### 2.4.2 方案二：手掌方向控制（自然）

```python
# 检测手掌倾斜角度，直接映射到模型旋转

def palm_to_rotation(landmarks):
    # 计算手掌法线向量
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    
    # 计算俯仰角和翻转角
    pitch = atan2(wrist.y - index_mcp.y, wrist.z - index_mcp.z)
    roll = atan2(pinky_mcp.x - index_mcp.x, pinky_mcp.y - index_mcp.y)
    
    return pitch, roll
```

#### 2.4.3 方案三：指尖追踪（精准）

```python
# 追踪食指指尖在屏幕上的移动轨迹

def fingertip_to_rotation(prev_landmark, curr_landmark):
    dx = curr_landmark.x - prev_landmark.x
    dy = curr_landmark.y - prev_landmark.y
    
    # 屏幕X移动 → 模型Y轴旋转
    # 屏幕Y移动 → 模型X轴旋转
    
    rotation_speed = 2.0  # 可调参数
    return dy * rotation_speed, dx * rotation_speed
```

**推荐：方案三（指尖追踪）**，因为：
1. 控制直观，用户体验好
2. 滑动方向自由
3. 易于添加速度感

## 3. 推荐技术栈

### 3.1 最终方案

| 组件 | 技术选型 | 版本要求 |
|------|---------|---------|
| 手势检测 | MediaPipe Hands | >=0.10 |
| 视频捕获 | OpenCV | >=4.5 |
| 3D渲染 | Pygame + PyOpenGL | pygame>=2.0, PyOpenGL>=3.1 |
| 数学计算 | NumPy | >=1.20 |

### 3.2 项目结构

```
mediapipe-test/
├── src/
│   ├── __init__.py
│   ├── hand_tracker.py      # 手势检测模块
│   ├── gesture_recognizer.py # 手势识别/滑动检测
│   ├── model_renderer.py    # 3D模型渲染
│   └── main.py              # 主程序入口
├── models/                   # 3D模型文件
├── research/                 # 研究文档
└── main.py                   # 入口文件
```

## 4. 实现步骤

### 步骤1：环境搭建
```bash
uv add mediapipe opencv-python pygame PyOpenGL numpy
```

### 步骤2：手部追踪模块
```python
import mediapipe as mp
import cv2

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def get_fingertip_position(self, frame):
        # 处理帧，返回指尖坐标
        # ...
```

### 步骤3：手势识别模块
```python
class GestureController:
    def __init__(self, sensitivity=2.0):
        self.prev_position = None
        self.sensitivity = sensitivity
    
    def detect_swipe(self, current_position):
        # 计算滑动方向和速度
        # 返回 rotation_x, rotation_y
        # ...
```

### 步骤4：3D渲染模块
```python
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

class ModelRenderer:
    def __init__(self):
        # 初始化Pygame和OpenGL
        # ...
    
    def rotate_model(self, dx, dy):
        # 根据手势更新模型旋转角度
        # ...
    
    def render(self):
        # 渲染3D模型
        # ...
```

### 步骤5：主程序集成
```python
def main():
    tracker = HandTracker()
    gesture = GestureController()
    renderer = ModelRenderer()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        landmarks = tracker.get_hand_landmarks(frame)
        
        if landmarks:
            rotation = gesture.detect_swipe(landmarks)
            renderer.rotate_model(*rotation)
        
        renderer.render()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

## 5. 性能优化建议

### 5.1 帧率优化
- 使用 `model_complexity=0` 降低计算量
- 设置 `image.flags.writeable = False` 避免不必要拷贝
- 使用 `cv2.flip()` 替代重新处理

### 5.2 旋转平滑
- 添加低通滤波器平滑手势输入
- 实现旋转惯性（释放后缓慢停止）
- 设置旋转速度和加速度限制

```python
# 平滑滤波示例
alpha = 0.3  # 平滑系数
smoothed_dx = alpha * raw_dx + (1 - alpha) * prev_dx
```

## 6. 参考资源

### 官方文档
- [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [MediaPipe Gesture Recognizer](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer)

### 示例项目
- [render-manipulation](https://github.com/yousifoa/render-manipulation) - 手势控制3D模型
- [3D-viewer-opengl-Python-Version](https://github.com/akashkumarsl/3D-viewer-opengl-Python-Version) - OpenCV + PyOpenGL 3D查看器

### 学习资料
- MediaPipe Python官方示例
- PyGame + PyOpenGL教程

## 7. 总结

本技术方案选择MediaPipe Hands作为手势检测方案，结合Pygame和PyOpenGL实现3D渲染。通过追踪指尖滑动方向和速度，可以实现直观、流畅的3D模型旋转控制。

推荐采用滑动检测方案，特点是：
1. **实现简单**：核心算法清晰
2. **体验自然**：用户直觉操作
3. **性能优秀**：30+ FPS流畅运行
4. **易于扩展**：可添加缩放、位移等功能
