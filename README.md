# vectorized-backtest

向量化回测引擎(雏形)。

**现在还没有真正的回测逻辑**——分组回测、IC、收益曲线这些都还没写。当前
阶段只是用来跑一次端到端验证:
[`sources`](https://github.com/20070316lbw-netizen/sources) 抓真实数据 →
[`load`](https://github.com/20070316lbw-netizen/load) 转 MultiIndex/存
parquet →
[`momfactor`](https://github.com/20070316lbw-netizen/momfactor) 算动量,
确认三个包接起来真的能跑, 环境变量之类的配置也顺手核对一遍。

跑数据用的脚本是 `src/vectorized_backtest/scratch.py`(未追踪, 见
`.gitignore` 里"开发用品"那行, 跟 `momfactor`/`load` 的同名约定一样),
跑出来的 parquet 落在仓库根目录的 `data/` 下(同样未追踪, 体积大也没必要
进 git, 随时能重新跑)。

## 开发

```bash
uv sync
uv run ruff check .
uv run pytest -v
```

目前没有正式模块和测试, `pytest` 跑起来是 0 collected, 属于正常情况;
等真正的回测逻辑(分组、IC、收益曲线)开始写, 再补测试。
