# input-replay

离线、无随机、无 `sleep` 的确定性输入录制与回放练习仓库。时间单位统一为毫秒，帧间隔由显式传入的 `frame_times` 决定。

## 题面

把玩家操作录下来回放时，结果每次都不一样：归档按固定 1/60 秒而不是实际可变帧间隔，事件常常错帧；落帧用 floor，边界上的事件掉进前一帧；去重把按下和抬起合并成一条，回放时长按失效；死区只减不除，推杆到头也到不了 1.0；设备切换不插边界事件，回放时用错设备状态；校验和把未排序的键也算进去，同内容得到不同哈希；回放按事件时间投递，同一帧内的事件会被重复投递；多轨合并按后到先得，低优先级设备覆盖高优先级；超长录制不截断，内存没有上限；时间戳回退不报错，回放顺序错乱。replay/recorder.py 是核心，tests/test_recorder.py 15个红。要修的是：可变帧率归档、边界落帧、按下-抬起去重、死区归一化、设备切换边界、校验和键排序、回放去重、多轨合并优先级、超长截断和时间戳回退校验。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始（未修复）状态必须恰好输出 `15 failed`，且没有 collection error、import error 或 syntax error。15 条测试都会调用 `replay/recorder.py` 中的桩，因此都应因 `NotImplementedError` 而失败；不要用 `-k`、`--ignore`、`skip` 等方式改变失败数量。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 其他 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 目录

```text
input-replay/
├── README.md
├── conftest.py
├── .gitignore
├── replay/
│   ├── __init__.py
│   └── recorder.py
└── tests/
    └── test_recorder.py
```

## 约束

- `tests/test_recorder.py` 不可修改；修复只能落在 `replay/recorder.py`，并保持 8 个公开函数的签名不变。
- 只使用标准库（`json`、`hashlib`、`bisect`、`math`）与 `pytest`；离线运行，不联网、不使用随机数、不调用 `sleep`。
- 时间戳必须显式传入，不能读取系统时钟。
