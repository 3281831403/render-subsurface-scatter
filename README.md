# render-subsurface-scatter

## 题面

给角色皮肤加次表面散射后，边缘发亮、手指还透不出红光：旧实现的扩散剖面没有归一化，能量随散射半径漂移；RGB 三通道共用同一个散射半径，皮肤偏色发灰；采样偏移全部落在正方向，高光整体偏亮且向一侧拖；模糊权重不随距离衰减，轮廓边缘反而被提亮；厚度越大越透，薄处发暗、厚处发光，和真实皮肤相反；菲涅尔反射用常数代替，掠射角没有高光；ior≤1 没有校验，菲涅尔计算里出现除零；albedo 越界不做钳制，颜色溢出到 0..1 之外；批量计算时所有材质共用一套采样，材质之间互相串扰；采样数 n=0 不校验，直接除零。sss/diffusion.py 是核心，tests/test_diffusion.py 15个红。要修的是：剖面归一化、按通道散射半径、居中采样、距离衰减、厚度衰减、角度菲涅尔、ior 校验、albedo 钳制、批量隔离和 n 校验。

## 运行方式

```bash
python -m pytest tests -q
```

初始状态输出恰好是 `15 failed`，且没有任何 collection error / import error / syntax error。把 `sss/diffusion.py` 的 8 个桩全部按 docstring 实现后，应变为 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- `tests/test_diffusion.py` 不可修改，评测以这 15 个测试为准。
- 只允许修改 `sss/diffusion.py`（必要时可补 `sss/__init__.py`）；只用标准库 `math` 与 pytest，不联网、不引入第三方依赖、不使用不可复现的随机性。
- 长度单位一律是毫米（mm），颜色分量是 `0..1` 的浮点数，数值容差 `1e-6`。材质统一为 `{"albedo": (r, g, b), "radius": float, "ior": float}`。
