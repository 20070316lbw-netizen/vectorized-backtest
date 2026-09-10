# vectorized-backtest

个人向量化回测引擎



## 开发

```bash
uv sync
uv run ruff check .
uv run pytest -v
```

目前没有正式模块和测试, `pytest` 跑起来是 0 collected, 属于正常情况;
等真正的回测逻辑(分组、IC、收益曲线)开始写, 再补测试。
